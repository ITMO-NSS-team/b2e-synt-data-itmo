"""Lifecycle tests — these encode the threat model's non-negotiables.

If any of these fail, the approval gate has become decorative.
"""
from __future__ import annotations

import pytest

from sim.registry import Registry, sha256_hex
from sim.skills import (
    LifecycleError, NotExecutable, SkillState, SkillStore, static_check,
)

GOOD_CODE = "import json\n\n\ndef run(payload):\n    return {'n': len(payload)}\n"


@pytest.fixture()
def store(tmp_path):
    registry = Registry(tmp_path / "r.db")
    yield SkillStore(registry)
    registry.close()


def _approve_and_activate(store, h, actor="alice"):
    store.transition(h, SkillState.PENDING_REVIEW, actor="agent")
    store.transition(h, SkillState.APPROVED, actor=actor, is_human_action=True)
    return store.transition(h, SkillState.ACTIVE, actor=actor, is_human_action=True)


# ------------------------------------------------------------------ entry


def test_agent_authored_skill_enters_as_draft(store):
    record = store.author(name="s", definition={"d": 1}, code=GOOD_CODE,
                          author="agent")
    assert record.state is SkillState.DRAFT


def test_uploaded_skill_enters_as_pending_review(store):
    record = store.upload(name="s", definition={"d": 1}, code=GOOD_CODE,
                          actor="alice")
    assert record.state is SkillState.PENDING_REVIEW


def test_author_has_no_state_parameter():
    """A state parameter is exactly the field an injected instruction would set."""
    import inspect
    from sim.skills import SkillStore as S
    assert "state" not in inspect.signature(S.author).parameters
    assert "state" not in inspect.signature(S.upload).parameters


# ------------------------------------------------------------- transitions


def test_draft_cannot_jump_straight_to_approved(store):
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    with pytest.raises(LifecycleError, match="not a permitted transition"):
        store.transition(h, SkillState.APPROVED, actor="alice", is_human_action=True)


def test_approval_requires_an_explicit_human_action(store):
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    store.transition(h, SkillState.PENDING_REVIEW, actor="agent")
    with pytest.raises(LifecycleError, match="requires an explicit human action"):
        store.transition(h, SkillState.APPROVED, actor="automation")


def test_activation_requires_an_explicit_human_action(store):
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    store.transition(h, SkillState.PENDING_REVIEW, actor="agent")
    store.transition(h, SkillState.APPROVED, actor="alice", is_human_action=True)
    with pytest.raises(LifecycleError):
        store.transition(h, SkillState.ACTIVE, actor="automation")


def test_retired_is_terminal(store):
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    _approve_and_activate(store, h)
    store.transition(h, SkillState.RETIRED, actor="alice", is_human_action=True)
    with pytest.raises(LifecycleError):
        store.transition(h, SkillState.ACTIVE, actor="alice", is_human_action=True)


# ------------------------------------------------------------- execution


def test_only_active_skills_execute(store):
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    with pytest.raises(NotExecutable, match="not active"):
        store.resolve_for_execution(h)
    _approve_and_activate(store, h)
    assert store.resolve_for_execution(h).decode() == GOOD_CODE


def test_unknown_hash_is_refused(store):
    with pytest.raises(NotExecutable, match="unknown skill hash"):
        store.resolve_for_execution("0" * 64)


def test_editing_an_approved_skill_does_not_inherit_approval(store):
    """The core content-addressing guarantee."""
    original = store.author(name="s", definition={}, code=GOOD_CODE, author="agent")
    _approve_and_activate(store, original.code_hash)

    evil = GOOD_CODE.replace("len(payload)", "0")
    edited = store.author(name="s", definition={}, code=evil, author="agent")

    assert edited.code_hash != original.code_hash
    assert edited.state is SkillState.DRAFT
    with pytest.raises(NotExecutable):
        store.resolve_for_execution(edited.code_hash)


def test_approve_then_swap_is_defeated_by_digest_recheck(store):
    """The attack the audit demonstrated: approve benign bytes under a name,
    then replace the bytes. Resolution is by hash and the digest is re-verified,
    so the swapped bytes have no approval to inherit."""
    good = store.author(name="attrition", definition={}, code=GOOD_CODE,
                        author="agent")
    _approve_and_activate(store, good.code_hash)

    evil_code = "def run(payload):\n    return {'owned': True}\n"
    evil = store.author(name="attrition", definition={}, code=evil_code,
                        author="agent")

    # Same NAME, active skill exists — but the evil hash is not executable.
    assert evil.name == good.name
    with pytest.raises(NotExecutable):
        store.resolve_for_execution(evil.code_hash)
    assert store.resolve_for_execution(good.code_hash).decode() == GOOD_CODE


def test_digest_mismatch_refuses_even_when_active(store):
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    _approve_and_activate(store, h)
    # Simulate tampering with the stored blob under an approved hash.
    tampered = b"def run(payload):\n    return {}\n"
    store.registry._conn.execute(
        "UPDATE objects SET body = ? WHERE hash = ?", (tampered, h))
    store.registry._conn.commit()
    with pytest.raises(NotExecutable, match="digest mismatch"):
        store.resolve_for_execution(h)


# ------------------------------------------------------------ static check


def test_static_check_never_executes_code():
    """A payload that would be catastrophic if exec'd is merely parsed."""
    import pathlib
    canary = pathlib.Path("/tmp/b2e-static-check-canary")
    canary.unlink(missing_ok=True)
    hostile = (f"import os\nos.system('touch {canary}')\n\n"
               f"def run(payload):\n    return {{}}\n")
    result = static_check(hostile)
    assert not canary.exists(), "static_check executed the module"
    assert "os" in result["imports"]
    assert "os" in result["undeclared"]


def test_static_check_requires_an_entrypoint():
    assert not static_check("x = 1\n")["ok"]
    assert static_check(GOOD_CODE)["ok"]


def test_static_check_reports_syntax_errors_without_raising():
    result = static_check("def run(:\n")
    assert not result["ok"]
    assert "syntax error" in result["error"]


# ------------------------------------------------------------------ audit


def test_every_transition_is_audited(store):
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    _approve_and_activate(store, h)
    actions = [e["action"] for e in store.registry.audit_read()]
    assert actions.count("skill.transition") == 3
    assert "skill.author" in actions


def test_registry_hash_changes_when_the_active_set_changes(store):
    before = store.registry_hash()
    h = store.author(name="s", definition={}, code=GOOD_CODE, author="agent").code_hash
    _approve_and_activate(store, h)
    assert store.registry_hash() != before


def test_identical_code_is_stored_once(store):
    a = store.author(name="s", definition={}, code=GOOD_CODE, author="agent")
    b = store.author(name="s", definition={}, code=GOOD_CODE, author="agent")
    assert a.code_hash == b.code_hash == sha256_hex(GOOD_CODE.encode())
