"""Foundation tests: run identity, latency model, config registry.

These three modules are load-bearing for every experiment. If the fingerprint can
be incomplete, comparisons are invalid; if latency is shape-blind, RQ2 is
unmeasurable; if config can be edited in place, old traces start lying.
"""
from __future__ import annotations

import statistics

import pytest

from sim import latency
from sim.fingerprint import (
    FINGERPRINT_FIELDS,
    IncompleteFingerprint,
    RunFingerprint,
)
from sim.registry import Registry, canonical_bytes, sha256_hex

GOOD = dict(
    agent_config_version="cfg@3",
    prompt_registry_version="system_prompt@7",
    skill_registry_hash="sha256:" + "ab" * 32,
    model_id="claude-haiku-4-5-20251001",
    temperature=0.0,
    data_snapshot_hash="heimdall-sandbox@78d53675db91e17f",
    traps_enabled=True,
    latency_profile="realistic",
)


# --------------------------------------------------------------- fingerprint


def test_fingerprint_accepts_complete_run():
    fp = RunFingerprint.create(**GOOD)
    assert len(fp.as_span_attributes()) == len(FINGERPRINT_FIELDS)
    assert all(k.startswith("b2e.run.") for k in fp.as_span_attributes())


@pytest.mark.parametrize("field", FINGERPRINT_FIELDS)
def test_fingerprint_refuses_any_missing_field(field):
    """The spec's enforcement point: refuse to start a run, not warn."""
    partial = {k: v for k, v in GOOD.items() if k != field}
    with pytest.raises(IncompleteFingerprint) as exc:
        RunFingerprint.create(**partial)
    assert field in exc.value.missing


def test_fingerprint_treats_blank_string_as_unset():
    """An unset env var arrives as "", and must fail like a missing key."""
    with pytest.raises(IncompleteFingerprint):
        RunFingerprint.create(**{**GOOD, "prompt_registry_version": "  "})


def test_fingerprint_rejects_unknown_latency_profile():
    with pytest.raises(ValueError):
        RunFingerprint.create(**{**GOOD, "latency_profile": "fast"})


def test_condition_id_distinguishes_the_rq1_variable():
    """Traps on and traps off must never collide: that is the RQ1 comparison."""
    on = RunFingerprint.create(**GOOD)
    off = RunFingerprint.create(**{**GOOD, "traps_enabled": False})
    assert on.condition_id != off.condition_id
    assert on.condition_id == RunFingerprint.create(**GOOD).condition_id


# ------------------------------------------------------------------ latency


def test_latency_is_reproducible_for_the_same_request():
    prof = latency.get_profile("realistic")
    req = {"model": "dm_core.employee_actual", "columns": ["grade"]}
    first = latency.sample_ms(prof, "mcp_query", request=req, n_columns=1, n_rows=10)
    again = latency.sample_ms(prof, "mcp_query", request=req, n_columns=1, n_rows=10)
    assert first == again


def test_repeat_calls_within_a_session_differ():
    prof = latency.get_profile("realistic")
    req = {"model": "x"}
    a = latency.sample_ms(prof, "mcp_query", request=req, occurrence=0)
    b = latency.sample_ms(prof, "mcp_query", request=req, occurrence=1)
    assert a != b


def test_wide_query_costs_more_than_narrow():
    """RQ2 depends on this: `columns: ["*"]` must be honestly more expensive."""
    prof = latency.get_profile("realistic")
    narrow = statistics.median(
        latency.sample_ms(prof, "mcp_query", request={"q": i}, n_columns=4, n_rows=50)
        for i in range(200))
    wide = statistics.median(
        latency.sample_ms(prof, "mcp_query", request={"q": i}, n_columns=642, n_rows=50)
        for i in range(200))
    assert wide > narrow * 3


def test_instant_profile_is_actually_free():
    prof = latency.get_profile("instant")
    assert latency.sample_ms(prof, "mcp_query", request={"q": 1}, n_columns=600) == 0.0


def test_degraded_widens_the_tail_not_just_the_median():
    """A loaded database degrades at p95 first. If degraded only scaled the
    median, the p95/p50 ratio would not move and the profile would teach the
    wrong lesson about sequential calls."""
    ratios = {}
    for name in ("realistic", "degraded"):
        prof = latency.get_profile(name)
        s = sorted(latency.sample_ms(prof, "mcp_query", request={"q": i},
                                     n_columns=8, n_rows=100) for i in range(400))
        ratios[name] = s[int(0.95 * len(s))] / statistics.median(s)
    assert ratios["degraded"] > ratios["realistic"]


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        latency.get_profile("turbo")


# ----------------------------------------------------------------- registry


@pytest.fixture()
def reg(tmp_path):
    r = Registry(tmp_path / "registry.db")
    yield r
    r.close()


def test_commit_appends_versions(reg):
    v1 = reg.commit("system_prompt", "prompt", {"text": "one"}, actor="alice")
    v2 = reg.commit("system_prompt", "prompt", {"text": "two"}, actor="bob")
    assert (v1.version, v2.version) == (1, 2)
    assert reg.head("system_prompt").version == 2
    assert [v.version for v in reg.history("system_prompt")] == [2, 1]


def test_old_versions_remain_fetchable_after_change(reg):
    """The whole point: a trace recorded against v1 can still resolve v1."""
    reg.commit("system_prompt", "prompt", {"text": "one"}, actor="alice")
    reg.commit("system_prompt", "prompt", {"text": "two"}, actor="alice")
    _, body = reg.load("system_prompt@1")
    assert body == {"text": "one"}


def test_identical_content_does_not_create_a_new_version(reg):
    """A no-op save must not manufacture a difference in the fingerprint."""
    a = reg.commit("cfg", "agent", {"temperature": 0.0}, actor="alice")
    b = reg.commit("cfg", "agent", {"temperature": 0.0}, actor="bob")
    assert a.version == b.version == 1


def test_key_order_does_not_change_the_hash():
    assert sha256_hex(canonical_bytes({"a": 1, "b": 2})) == \
           sha256_hex(canonical_bytes({"b": 2, "a": 1}))


def test_edit_changes_the_hash(reg):
    """Approval is pinned to the hash, so an edit must not inherit it."""
    v1 = reg.commit("skill.headcount", "skill", {"code": "return 1"}, actor="agent")
    v2 = reg.commit("skill.headcount", "skill", {"code": "return 2"}, actor="agent")
    assert v1.hash != v2.hash


def test_resolve_head_and_pinned(reg):
    reg.commit("cfg", "agent", {"n": 1}, actor="a")
    reg.commit("cfg", "agent", {"n": 2}, actor="a")
    assert reg.resolve("cfg").version == 2
    assert reg.resolve("cfg@1").version == 1
    assert reg.resolve("cfg@99") is None
    assert reg.resolve("nope") is None


def test_every_commit_lands_in_the_audit_log(reg):
    reg.commit("cfg", "agent", {"n": 1}, actor="alice", note="initial")
    entries = reg.audit_read()
    assert len(entries) == 1
    assert entries[0]["actor"] == "alice"
    assert entries[0]["action"] == "config.commit"
    assert entries[0]["target"] == "cfg@1"


def test_audit_log_is_append_only_in_practice(reg):
    reg.commit("cfg", "agent", {"n": 1}, actor="alice")
    reg.audit_write(actor="bob", action="skill.approve", target="skill.x",
                    detail={"hash": "abc"})
    ids = [e["id"] for e in reg.iter_audit()]
    assert ids == sorted(ids)
    assert len(ids) == 2


def test_registry_survives_reopen(tmp_path):
    path = tmp_path / "r.db"
    r1 = Registry(path)
    r1.commit("cfg", "agent", {"n": 1}, actor="alice")
    r1.close()
    r2 = Registry(path)
    try:
        assert r2.head("cfg").version == 1
        assert r2.load("cfg@1")[1] == {"n": 1}
    finally:
        r2.close()
