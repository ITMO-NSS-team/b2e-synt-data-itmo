"""Skill lifecycle: draft → pending_review → approved → active → retired.

Implements the state machine specified in
``docs/skill-execution-threat-model.md`` §3. The rules that are not negotiable:

* Agent-authored skills **always** enter as ``draft``. There is no parameter
  anywhere in the agent's tool surface that can set a state.
* Uploaded skills **always** enter as ``pending_review``.
* Only an explicit human action promotes to ``approved``.
* Approval is pinned to the **SHA-256 of the exact bytes**, never to the name.
* Only ``active`` is executable.

The name/hash distinction is the one that stops a real attack. The audit found a
working approve-then-swap: get a benign ``attrition_by_dept`` approved, then
write different bytes for that name, and a name-resolving executor runs the
swapped code with no re-approval. Resolution is therefore by hash, and the
runner re-verifies the digest of what it is about to execute.
"""
from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sim.registry import Registry, canonical_bytes, sha256_hex


class SkillState(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    ACTIVE = "active"
    RETIRED = "retired"
    REJECTED = "rejected"


#: Allowed transitions. Anything absent is refused, including "approved" from
#: "draft" without passing review, and any resurrection of a retired hash.
TRANSITIONS: dict[SkillState, frozenset[SkillState]] = {
    SkillState.DRAFT: frozenset({SkillState.PENDING_REVIEW, SkillState.REJECTED}),
    SkillState.PENDING_REVIEW: frozenset({SkillState.APPROVED, SkillState.REJECTED}),
    SkillState.APPROVED: frozenset({SkillState.ACTIVE, SkillState.RETIRED}),
    SkillState.ACTIVE: frozenset({SkillState.APPROVED, SkillState.RETIRED}),
    SkillState.RETIRED: frozenset(),
    SkillState.REJECTED: frozenset(),
}

#: Transitions a human must perform. The agent may submit a draft for review;
#: it may never approve or activate.
HUMAN_ONLY = frozenset({SkillState.APPROVED, SkillState.ACTIVE})

#: Imports a skill may declare. Enforced at runtime by the sandbox, and checked
#: statically here so a reviewer sees the capability surface up front. This list
#: is telemetry and review-aid, NOT a security boundary — see the threat model.
DECLARABLE_IMPORTS = frozenset({
    "json", "math", "statistics", "datetime", "decimal", "fractions", "re",
    "collections", "itertools", "functools", "operator", "typing",
    "dataclasses", "enum", "uuid", "hashlib", "textwrap", "unicodedata",
})


class LifecycleError(RuntimeError):
    pass


class NotExecutable(RuntimeError):
    """The runner refused: hash unknown, not active, or digest mismatch."""


@dataclass(frozen=True, slots=True)
class SkillRecord:
    name: str
    code_hash: str
    definition_hash: str
    state: SkillState
    author: str
    created_at: float
    updated_at: float

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "code_hash": self.code_hash,
                "definition_hash": self.definition_hash, "state": self.state.value,
                "author": self.author, "created_at": self.created_at,
                "updated_at": self.updated_at}


SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    code_hash       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    definition_hash TEXT NOT NULL,
    state           TEXT NOT NULL,
    author          TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS skills_by_name ON skills(name);
CREATE INDEX IF NOT EXISTS skills_by_state ON skills(state);
"""


def static_check(code: str) -> dict[str, Any]:
    """Inspect draft code **without executing any of it**.

    Uses ``ast.parse`` only. ``exec``, ``eval``, ``__import__`` and ``importlib``
    are never applied to draft code, and neither is any "helpful" preview, lint,
    or dry-run. That prohibition is the single most likely way this design gets
    defeated later, so the check that would tempt someone into it lives here, in
    a form that provably cannot run anything.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "error": f"syntax error line {exc.lineno}: {exc.msg}",
                "imports": [], "undeclared": []}

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    has_entrypoint = any(
        isinstance(node, ast.FunctionDef) and node.name == "run"
        for node in tree.body)

    return {
        "ok": has_entrypoint,
        "error": "" if has_entrypoint else "no top-level `def run(payload)` entrypoint",
        "imports": sorted(imports),
        "undeclared": sorted(imports - DECLARABLE_IMPORTS),
        "lines": code.count("\n") + 1,
    }


class SkillStore:
    """Lifecycle store. Content lives in the registry; state lives here."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.registry._conn.executescript(SCHEMA)
        self.registry._conn.commit()

    # ------------------------------------------------------------- creation

    def author(self, *, name: str, definition: dict[str, Any], code: str,
               author: str) -> SkillRecord:
        """Agent-authored. Always lands in ``draft``, no exceptions.

        There is no ``state`` parameter. That is not an oversight — a state
        parameter is exactly the field an injected instruction would set.
        """
        return self._create(name=name, definition=definition, code=code,
                            author=author, state=SkillState.DRAFT,
                            action="skill.author")

    def upload(self, *, name: str, definition: dict[str, Any], code: str,
               actor: str) -> SkillRecord:
        """Human-uploaded. Lands in ``pending_review`` — upload is not approval."""
        return self._create(name=name, definition=definition, code=code,
                            author=actor, state=SkillState.PENDING_REVIEW,
                            action="skill.upload")

    def _create(self, *, name: str, definition: dict[str, Any], code: str,
                author: str, state: SkillState, action: str) -> SkillRecord:
        code_bytes = code.encode("utf-8")
        code_hash = sha256_hex(code_bytes)
        definition_hash = sha256_hex(canonical_bytes(definition))

        self.registry.put_object("skill_code", code_bytes)
        self.registry.put_object("skill_definition", definition)

        existing = self.get(code_hash)
        if existing is not None:
            return existing

        now = time.time()
        with self.registry._lock:
            self.registry._conn.execute(
                "INSERT INTO skills(code_hash, name, definition_hash, state, "
                "author, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (code_hash, name, definition_hash, state.value, author, now, now))
            self.registry._conn.commit()
        self.registry.audit_write(
            actor=author, action=action, target=f"{name}@{code_hash[:12]}",
            detail={"state": state.value, "code_hash": code_hash,
                    "definition_hash": definition_hash,
                    "static_check": static_check(code)})
        return self.get(code_hash)                              # type: ignore[return-value]

    # ------------------------------------------------------------ transition

    def transition(self, code_hash: str, to: SkillState, *, actor: str,
                   is_human_action: bool = False, note: str = "") -> SkillRecord:
        record = self.get(code_hash)
        if record is None:
            raise LifecycleError(f"no skill with hash {code_hash[:12]}…")

        allowed = TRANSITIONS[record.state]
        if to not in allowed:
            raise LifecycleError(
                f"{record.state.value} → {to.value} is not a permitted transition "
                f"(allowed: {sorted(s.value for s in allowed) or 'none — terminal'})")

        if to in HUMAN_ONLY and not is_human_action:
            raise LifecycleError(
                f"promotion to {to.value} requires an explicit human action in the "
                f"admin UI. Automated approval would make the approval gate "
                f"decorative, which the threat model treats as a defect.")

        now = time.time()
        with self.registry._lock:
            self.registry._conn.execute(
                "UPDATE skills SET state = ?, updated_at = ? WHERE code_hash = ?",
                (to.value, now, code_hash))
            self.registry._conn.commit()
        self.registry.audit_write(
            actor=actor, action="skill.transition",
            target=f"{record.name}@{code_hash[:12]}",
            detail={"from": record.state.value, "to": to.value,
                    "code_hash": code_hash, "human": is_human_action, "note": note})
        return self.get(code_hash)                              # type: ignore[return-value]

    # --------------------------------------------------------------- reading

    def get(self, code_hash: str) -> SkillRecord | None:
        row = self.registry._conn.execute(
            "SELECT * FROM skills WHERE code_hash = ?", (code_hash,)).fetchone()
        if row is None:
            return None
        return SkillRecord(
            name=row["name"], code_hash=row["code_hash"],
            definition_hash=row["definition_hash"],
            state=SkillState(row["state"]), author=row["author"],
            created_at=row["created_at"], updated_at=row["updated_at"])

    def list(self, state: SkillState | None = None) -> list[SkillRecord]:
        if state is None:
            rows = self.registry._conn.execute(
                "SELECT * FROM skills ORDER BY updated_at DESC").fetchall()
        else:
            rows = self.registry._conn.execute(
                "SELECT * FROM skills WHERE state = ? ORDER BY updated_at DESC",
                (state.value,)).fetchall()
        return [SkillRecord(
            name=r["name"], code_hash=r["code_hash"],
            definition_hash=r["definition_hash"], state=SkillState(r["state"]),
            author=r["author"], created_at=r["created_at"],
            updated_at=r["updated_at"]) for r in rows]

    def code(self, code_hash: str) -> bytes | None:
        return self.registry.get_object(code_hash)

    def active_hashes(self) -> list[str]:
        return [r.code_hash for r in self.list(SkillState.ACTIVE)]

    def registry_hash(self) -> str:
        """Digest over the active set, for the run fingerprint."""
        return "sha256:" + sha256_hex(canonical_bytes(sorted(self.active_hashes())))

    # ------------------------------------------------------------ execution

    def resolve_for_execution(self, code_hash: str) -> bytes:
        """The only path from a hash to executable bytes.

        Three refusals, in order: unknown hash, not active, digest mismatch. The
        third is not redundant — it is what defeats approve-then-swap, because
        the bytes are re-hashed immediately before they are handed to the runner.
        """
        record = self.get(code_hash)
        if record is None:
            raise NotExecutable(f"unknown skill hash {code_hash[:12]}…")
        if record.state is not SkillState.ACTIVE:
            raise NotExecutable(
                f"skill {record.name} is {record.state.value}, not active; "
                f"only active skills execute")
        code = self.code(code_hash)
        if code is None:
            raise NotExecutable(f"no stored bytes for {code_hash[:12]}…")
        actual = sha256_hex(code)
        if actual != code_hash:
            raise NotExecutable(
                f"digest mismatch: approved {code_hash[:12]}… but stored bytes "
                f"hash to {actual[:12]}…. Refusing to execute.")
        return code


def approval_manifest(store: SkillStore) -> str:
    """What the sandbox is allowed to run, as JSON. Written by approval only."""
    return json.dumps(
        {r.code_hash: {"name": r.name, "state": r.state.value}
         for r in store.list(SkillState.ACTIVE)},
        sort_keys=True, ensure_ascii=False)
