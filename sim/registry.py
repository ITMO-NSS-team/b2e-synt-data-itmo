"""Content-addressed, append-only registry for everything that is configuration.

Design rule: nothing is edited in place
---------------------------------------
Two runs are only comparable if you can still fetch, byte for byte, the
configuration each one used. An in-place edit destroys that retroactively — every
trace recorded before the edit now points at a system prompt that no longer
exists, and the fingerprint that names it becomes a lie. So a "change" here is
always an append: a new version row pointing at a new immutable blob.

Content addressing carries a second job, from the threat model. An approved skill
is pinned by the SHA-256 of its content. Approval attaches to the *hash*, not to
the name, so editing an approved skill cannot silently inherit its approval — the
edit produces a different hash, which has no approval record.

Storage is SQLite. It is a single file, needs no container, survives restarts,
and on a 2 vCPU / 4 GiB box that matters more than concurrency headroom we will
never use. WAL mode so a reader (the agent) is never blocked by a writer (the
admin UI).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
-- Immutable blobs, keyed by SHA-256 of the exact bytes stored.
CREATE TABLE IF NOT EXISTS objects (
    hash        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    body        BLOB NOT NULL,
    created_at  REAL NOT NULL
);

-- Named history. One row per version; rows are never updated or deleted.
CREATE TABLE IF NOT EXISTS versions (
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    hash        TEXT NOT NULL REFERENCES objects(hash),
    actor       TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE INDEX IF NOT EXISTS versions_by_name ON versions(name, version DESC);

-- Append-only audit log. Every mutation anywhere in the system lands here.
CREATE TABLE IF NOT EXISTS audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    target      TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS audit_by_ts ON audit(ts DESC);
CREATE INDEX IF NOT EXISTS audit_by_target ON audit(target, ts DESC);
"""

#: Guard rails against an append-only store being quietly turned into a mutable
#: one by a later edit. Any UPDATE/DELETE on these tables is a bug.
APPEND_ONLY_TABLES = ("objects", "versions", "audit")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON encoding, so logically identical config hashes identically.

    Sorted keys and fixed separators: without this, two configs differing only in
    key order would produce two hashes, two "versions", and a spurious difference
    between runs that are in fact the same condition.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class Version:
    name: str
    kind: str
    version: int
    hash: str
    actor: str
    note: str
    created_at: float

    @property
    def ref(self) -> str:
        """Human-facing version label used in fingerprints, e.g. ``system_prompt@7``."""
        return f"{self.name}@{self.version}"


class Registry:
    """Append-only config store. Thread-safe; safe across processes via WAL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------- blobs

    def put_object(self, kind: str, value: Any) -> str:
        """Store a blob, return its hash. Idempotent: same content, same hash."""
        body = value if isinstance(value, bytes) else canonical_bytes(value)
        digest = sha256_hex(body)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO objects(hash, kind, body, created_at) "
                "VALUES (?,?,?,?)",
                (digest, kind, body, time.time()),
            )
            self._conn.commit()
        return digest

    def get_object(self, digest: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT body FROM objects WHERE hash = ?", (digest,)
        ).fetchone()
        return None if row is None else bytes(row["body"])

    def get_json(self, digest: str) -> Any | None:
        body = self.get_object(digest)
        return None if body is None else json.loads(body.decode("utf-8"))

    # -------------------------------------------------------------- versions

    def commit(self, name: str, kind: str, value: Any, *, actor: str,
               note: str = "") -> Version:
        """Append a new version of ``name``.

        If the content is byte-identical to the current head, no new version is
        created and the existing head is returned. Recording "version 8, same as
        version 7" would put a difference in the fingerprint where there is none
        in the system, and every A/B comparison across that boundary would look
        like it varied something when it did not.
        """
        digest = self.put_object(kind, value)
        with self._lock:
            head = self._conn.execute(
                "SELECT * FROM versions WHERE name = ? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
            if head is not None and head["hash"] == digest:
                return _as_version(head)

            nxt = 1 if head is None else int(head["version"]) + 1
            now = time.time()
            self._conn.execute(
                "INSERT INTO versions(name, kind, version, hash, actor, note, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (name, kind, nxt, digest, actor, note, now),
            )
            self._conn.execute(
                "INSERT INTO audit(ts, actor, action, target, detail) VALUES (?,?,?,?,?)",
                (now, actor, "config.commit", f"{name}@{nxt}",
                 json.dumps({"kind": kind, "hash": digest, "note": note})),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM versions WHERE name = ? AND version = ?", (name, nxt)
            ).fetchone()
        return _as_version(row)

    def head(self, name: str) -> Version | None:
        row = self._conn.execute(
            "SELECT * FROM versions WHERE name = ? ORDER BY version DESC LIMIT 1",
            (name,),
        ).fetchone()
        return None if row is None else _as_version(row)

    def get_version(self, name: str, version: int) -> Version | None:
        row = self._conn.execute(
            "SELECT * FROM versions WHERE name = ? AND version = ?", (name, version)
        ).fetchone()
        return None if row is None else _as_version(row)

    def resolve(self, ref: str) -> Version | None:
        """Resolve ``name`` (head) or ``name@N`` (pinned) to a version."""
        if "@" in ref:
            name, _, raw = ref.rpartition("@")
            try:
                return self.get_version(name, int(raw))
            except ValueError:
                return None
        return self.head(ref)

    def history(self, name: str) -> list[Version]:
        rows = self._conn.execute(
            "SELECT * FROM versions WHERE name = ? ORDER BY version DESC", (name,)
        ).fetchall()
        return [_as_version(r) for r in rows]

    def names(self, kind: str | None = None) -> list[str]:
        if kind is None:
            rows = self._conn.execute(
                "SELECT DISTINCT name FROM versions ORDER BY name").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT name FROM versions WHERE kind = ? ORDER BY name",
                (kind,)).fetchall()
        return [r["name"] for r in rows]

    def load(self, ref: str) -> tuple[Version, Any]:
        """Resolve a ref and return its parsed content. Raises if absent."""
        version = self.resolve(ref)
        if version is None:
            raise KeyError(f"no such config ref: {ref!r}")
        value = self.get_json(version.hash)
        if value is None:
            raise KeyError(f"config {ref!r} points at missing object {version.hash}")
        return version, value

    # ----------------------------------------------------------------- audit

    def audit_write(self, *, actor: str, action: str, target: str,
                    detail: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit(ts, actor, action, target, detail) VALUES (?,?,?,?,?)",
                (time.time(), actor, action, target,
                 json.dumps(detail or {}, ensure_ascii=False)),
            )
            self._conn.commit()

    def audit_read(self, *, target: str | None = None,
                   limit: int = 200) -> list[dict[str, Any]]:
        if target is None:
            rows = self._conn.execute(
                "SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM audit WHERE target = ? ORDER BY ts DESC LIMIT ?",
                (target, limit)).fetchall()
        return [
            {"id": r["id"], "ts": r["ts"], "actor": r["actor"], "action": r["action"],
             "target": r["target"], "detail": json.loads(r["detail"])}
            for r in rows
        ]

    def iter_audit(self) -> Iterator[dict[str, Any]]:
        for row in self._conn.execute("SELECT * FROM audit ORDER BY id"):
            yield {"id": row["id"], "ts": row["ts"], "actor": row["actor"],
                   "action": row["action"], "target": row["target"],
                   "detail": json.loads(row["detail"])}


def _as_version(row: sqlite3.Row) -> Version:
    return Version(name=row["name"], kind=row["kind"], version=int(row["version"]),
                   hash=row["hash"], actor=row["actor"], note=row["note"],
                   created_at=float(row["created_at"]))
