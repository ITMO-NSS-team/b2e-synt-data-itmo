"""Session and job storage for the agent service.

SQLite, for the same reason the config registry uses it: one file, no extra
container, and on a 2 vCPU / 3.8 GiB box that matters more than concurrency
headroom this environment will never need.

Idempotency keys are stored with their result, not just as a seen-set. A retried
batch submission must return *the original job*, not merely be refused — a
researcher whose connection dropped needs the job id, and refusing them leaves
them unable to find the work that is already running.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    employee_id    TEXT NOT NULL,
    config_ref     TEXT NOT NULL,
    created_at     REAL NOT NULL,
    fingerprint    TEXT NOT NULL,
    metadata       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS sessions_by_employee ON sessions(employee_id, created_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    seq          INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    trace_id     TEXT,
    created_at   REAL NOT NULL,
    stats        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id, seq);

CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    idempotency_key  TEXT UNIQUE,
    status           TEXT NOT NULL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    request          TEXT NOT NULL,
    projection       TEXT NOT NULL DEFAULT '{}',
    result           TEXT NOT NULL DEFAULT '{}',
    error            TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs(status, created_at DESC);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -------------------------------------------------------------- sessions

    def create_session(self, *, employee_id: str, config_ref: str,
                       fingerprint: dict[str, Any],
                       metadata: dict[str, Any] | None = None) -> str:
        session_id = f"ses_{uuid.uuid4().hex[:20]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(id, employee_id, config_ref, created_at, "
                "fingerprint, metadata) VALUES (?,?,?,?,?,?)",
                (session_id, str(employee_id), config_ref, time.time(),
                 json.dumps(fingerprint), json.dumps(metadata or {})))
            self._conn.commit()
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "employee_id": row["employee_id"],
                "config_ref": row["config_ref"], "created_at": row["created_at"],
                "fingerprint": json.loads(row["fingerprint"]),
                "metadata": json.loads(row["metadata"])}

    def list_sessions(self, *, employee_id: str | None = None,
                      limit: int = 50) -> list[dict[str, Any]]:
        if employee_id:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE employee_id = ? "
                "ORDER BY created_at DESC LIMIT ?", (str(employee_id), limit)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"id": r["id"], "employee_id": r["employee_id"],
                 "config_ref": r["config_ref"], "created_at": r["created_at"],
                 "fingerprint": json.loads(r["fingerprint"])} for r in rows]

    # -------------------------------------------------------------- messages

    def append_message(self, *, session_id: str, role: str, content: str,
                       trace_id: str | None = None,
                       stats: dict[str, Any] | None = None) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS s FROM messages WHERE session_id = ?",
                (session_id,)).fetchone()
            seq = int(row["s"]) + 1
            self._conn.execute(
                "INSERT INTO messages(session_id, seq, role, content, trace_id, "
                "created_at, stats) VALUES (?,?,?,?,?,?,?)",
                (session_id, seq, role, content, trace_id, time.time(),
                 json.dumps(stats or {}, ensure_ascii=False)))
            self._conn.commit()
        return seq

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,)).fetchall()
        return [{"seq": r["seq"], "role": r["role"], "content": r["content"],
                 "trace_id": r["trace_id"], "created_at": r["created_at"],
                 "stats": json.loads(r["stats"])} for r in rows]

    def history_for_model(self, session_id: str) -> list[dict[str, Any]]:
        """Prior turns as plain user/assistant text.

        Tool-call blocks are deliberately not replayed across turns: they are in
        the trace, but re-sending them would grow the window with material the
        model already summarised into its answer.
        """
        return [{"role": m["role"], "content": m["content"]}
                for m in self.messages(session_id)
                if m["role"] in ("user", "assistant")]

    # ------------------------------------------------------------------ jobs

    def create_job(self, *, request: dict[str, Any],
                   idempotency_key: str | None) -> tuple[str, bool]:
        """Returns (job_id, created). ``created=False`` means the key was reused."""
        if idempotency_key:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,)).fetchone()
            if row is not None:
                return row["id"], False

        job_id = f"exp_{uuid.uuid4().hex[:20]}"
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO jobs(id, idempotency_key, status, created_at, "
                    "updated_at, request) VALUES (?,?,?,?,?,?)",
                    (job_id, idempotency_key, "queued", now, now,
                     json.dumps(request, ensure_ascii=False)))
                self._conn.commit()
            except sqlite3.IntegrityError:
                # Lost a race on the unique key: return the winner's job.
                row = self._conn.execute(
                    "SELECT id FROM jobs WHERE idempotency_key = ?",
                    (idempotency_key,)).fetchone()
                if row is not None:
                    return row["id"], False
                raise
        return job_id, True

    def update_job(self, job_id: str, *, status: str | None = None,
                   projection: dict[str, Any] | None = None,
                   result: dict[str, Any] | None = None,
                   error: str | None = None) -> None:
        sets, values = ["updated_at = ?"], [time.time()]
        if status is not None:
            sets.append("status = ?"); values.append(status)
        if projection is not None:
            sets.append("projection = ?"); values.append(json.dumps(projection))
        if result is not None:
            sets.append("result = ?")
            values.append(json.dumps(result, ensure_ascii=False, default=str))
        if error is not None:
            sets.append("error = ?"); values.append(error)
        values.append(job_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", values)
            self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "status": row["status"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "request": json.loads(row["request"]),
                "projection": json.loads(row["projection"]),
                "result": json.loads(row["result"]), "error": row["error"]}

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, status, created_at, updated_at FROM jobs "
            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
