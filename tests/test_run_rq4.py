"""The RQ4 driver, exercised against a fake agent endpoint — never a live
model. Covers the five properties the plan calls out explicitly: resume
skips completed work; a 429 retries then continues; arms interleave; the
schedule replays identically across arms; a failed turn is recorded rather
than aborting the run.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b2e.build import build                                    # noqa: E402
from sim.oracle.labels import GoldLabels                       # noqa: E402
from sim.registry import Registry                                # noqa: E402

_SPEC = importlib.util.spec_from_file_location("run_rq4", ROOT / "scripts" / "run_rq4.py")
run_rq4 = importlib.util.module_from_spec(_SPEC)
sys.modules["run_rq4"] = run_rq4          # registered before exec: dataclasses
_SPEC.loader.exec_module(run_rq4)         # need this to resolve deferred annotations

CATALOG = ROOT / "catalog" / "snapshot.json"
DATA = ROOT / ".pytest-run-rq4"
SEED = 424242
HR_EMPLOYEE_ID = "9999999"


@pytest.fixture(scope="module")
def gold():
    shutil.rmtree(DATA, ignore_errors=True)
    build(SEED, 800, DATA, CATALOG, progress=False)
    yield GoldLabels(DATA)
    shutil.rmtree(DATA, ignore_errors=True)


# ------------------------------------------------------------------- fakes


_ANSWER_BLOCK = "Ответ.\n```answer\nrefused: true\nreason: тест\n```"


class _FakePhoenix:
    """No spans, ever — forces every turn through the `_facts_from_stats`
    fallback so the fake agent's `stats` payload is what gets scored."""

    def spans_for_session(self, session_id: str, *, limit: int = 1000) -> list:
        return []


def _tool_span() -> dict:
    """One minimal real-shaped TOOL span — see test_reflection_extract.py's
    `_span`/`_tool_span` fixtures, whose shape this mirrors, for what
    `call_records` actually expects (Phoenix's flattened REST attributes)."""
    return {
        "context": {"span_id": "s1", "trace_id": "t1"}, "parent_id": None,
        "name": "tool.mcp__heimdall__mcp_query", "span_kind": "TOOL",
        "start_time": "2026-08-09T05:16:48.000000+00:00",
        "end_time": "2026-08-09T05:16:49.000000+00:00",
        "attributes": {
            "input.value": json.dumps({"schema": "dm_core", "logic_model": "employee_actual",
                                       "columns": ["grade_level"]}),
            "input.mime_type": "application/json",
            "tool.name": "mcp__heimdall__mcp_query",
        },
    }


class _SpanfulPhoenix:
    """Returns one real TOOL span for every session — proves run_one_turn
    actually threads call_records() output through to the reflection episode
    rather than always handing reflection an empty call list."""

    def spans_for_session(self, session_id: str, *, limit: int = 1000) -> list:
        return [_tool_span()]


def make_agent_handler(*, fail_status: int = 500, fail_when: "callable" = None,
                       fail_budget: dict | None = None):
    """A fake ``b2e-agent`` service. ``fail_when(question_text) -> bool``
    marks a question as one that should fail; ``fail_budget`` (keyed by
    question text prefix) counts down remaining failures before it starts
    succeeding — the 429-then-recovers case.
    """
    fail_budget = fail_budget or {}
    seen_sessions: dict[str, str] = {}
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/sessions":
            payload = json.loads(request.content)
            sid = f"sess-{counter['n']}"
            counter["n"] += 1
            seen_sessions[sid] = payload["employee_id"]
            return httpx.Response(201, json={
                "session_id": sid, "employee_id": payload["employee_id"],
                "fingerprint": {"condition_id": "cond-1"}, "condition_id": "cond-1"})
        if request.method == "POST" and path.endswith("/messages"):
            payload = json.loads(request.content)
            text = payload["content"]
            key = next((k for k in fail_budget if text.startswith(k)), None)
            if key is not None and fail_budget[key] > 0:
                fail_budget[key] -= 1
                return httpx.Response(429, json={"detail": "rate limited"})
            if fail_when is not None and fail_when(text):
                return httpx.Response(fail_status, json={"detail": "boom"})
            sid = path.split("/")[2]
            return httpx.Response(200, json={
                "session_id": sid, "answer": _ANSWER_BLOCK, "trace_id": "trace-1",
                "stop_reason": "end_turn",
                "stats": {"iterations": 1, "tool_calls": 1, "heimdall_calls": 1,
                         "prompt_tokens": 100, "completion_tokens": 50,
                         "total_tokens": 150, "cost_usd": 0.001},
                "errors": [], "condition_id": "cond-1"})
        return httpx.Response(404, json={"detail": "not found"})

    return handler


def make_cfg(tmp_path, *, transport, epochs=1, concurrency=3,
            curator_client=None) -> "run_rq4.RunConfig":
    return run_rq4.RunConfig(
        seed=SEED, replications=1, epochs=epochs, concurrency=concurrency,
        hr_employee_id=HR_EMPLOYEE_ID, run_id="test", out_dir=tmp_path / "out",
        dry_run=False, registry_db=str(tmp_path / "registry.db"),
        transport=transport, retry_sleep=lambda s: None,
        curator_client_override=curator_client or run_rq4.NullCuratorClient(),
        phoenix_override=_FakePhoenix(),
    )


# ----------------------------------------------------------- AgentClient unit


def test_429_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(201, json={"session_id": "s1"})

    client = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    agent = run_rq4.AgentClient(client=client, sleep=lambda s: None)

    result, attempts = agent.create_session(employee_id="1", config_ref="agent_config",
                                            metadata={})
    assert result == {"session_id": "s1"}
    assert attempts == 3
    assert calls["n"] == 3


def test_persistent_failure_raises_dispatch_failed_after_max_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "down"})

    client = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    agent = run_rq4.AgentClient(client=client, sleep=lambda s: None, max_attempts=3)

    with pytest.raises(run_rq4.DispatchFailed) as excinfo:
        agent.create_session(employee_id="1", config_ref="agent_config", metadata={})
    assert excinfo.value.attempts == 3


def test_non_retryable_status_fails_fast():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"detail": "bad request"})

    client = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    agent = run_rq4.AgentClient(client=client, sleep=lambda s: None, max_attempts=6)

    with pytest.raises(run_rq4.DispatchFailed):
        agent.create_session(employee_id="1", config_ref="agent_config", metadata={})
    assert calls["n"] == 1  # never retried a 422


# --------------------------------------------------------------- interleaving


def test_arms_interleave_never_arm_by_arm(gold):
    items = run_rq4.build_schedule(gold, seed=SEED, replication=1,
                                   hr_employee_id=HR_EMPLOYEE_ID)
    by_epoch: dict[int, list] = {}
    for it in items:
        by_epoch.setdefault(it.epoch, []).append(it)

    work = run_rq4.interleaved_epoch_work(by_epoch[0], run_rq4.ARMS, eff_seed=1, epoch=0)
    assert len(work) == 30 * len(run_rq4.ARMS)
    # No two consecutive dispatches share an arm — the strong form of "never
    # arm-by-arm": a schedule that went even a single arm-by-arm stretch of 2
    # would fail this.
    for (arm_a, _), (arm_b, _) in zip(work, work[1:]):
        assert arm_a != arm_b


def test_interleaving_covers_every_item_under_every_arm(gold):
    items = run_rq4.build_schedule(gold, seed=SEED, replication=1,
                                   hr_employee_id=HR_EMPLOYEE_ID)
    by_epoch: dict[int, list] = {}
    for it in items:
        by_epoch.setdefault(it.epoch, []).append(it)

    work = run_rq4.interleaved_epoch_work(by_epoch[2], run_rq4.ARMS, eff_seed=1, epoch=2)
    seen = {(arm, item.question_id) for arm, item in work}
    assert len(seen) == 30 * len(run_rq4.ARMS)
    for item in by_epoch[2]:
        for arm in run_rq4.ARMS:
            assert (arm, item.question_id) in seen


# -------------------------------------------------------------------- run()


def test_schedule_replays_identically_across_arms(tmp_path, gold):
    transport = httpx.MockTransport(make_agent_handler())
    cfg = make_cfg(tmp_path, transport=transport)
    run_rq4.run(cfg, gold=gold)

    rows = _read_rows(cfg.out_dir / "turns.jsonl")
    by_question: dict[str, list[dict]] = {}
    for r in rows:
        by_question.setdefault(r["question_id"], []).append(r)

    assert len(by_question) == 30  # epoch 0 only
    for qid, arm_rows in by_question.items():
        assert {r["arm"] for r in arm_rows} == set(run_rq4.ARMS)
        pair_ids = {r["pair_id"] for r in arm_rows}
        categories = {r["category"] for r in arm_rows}
        classes = {r["question_class"] for r in arm_rows}
        employees = {r["employee_id"] for r in arm_rows}
        assert len(pair_ids) == 1, qid
        assert len(categories) == 1, qid
        assert len(classes) == 1, qid
        assert len(employees) == 1, qid  # same acting identity in every arm


def test_a_failed_turn_is_recorded_not_aborting(tmp_path, gold):
    items = run_rq4.build_schedule(gold, seed=SEED, replication=1,
                                   hr_employee_id=HR_EMPLOYEE_ID)
    epoch0 = [it for it in items if it.epoch == 0]
    poison = epoch0[0].text[:30]

    handler = make_agent_handler(fail_when=lambda text: text.startswith(poison))
    transport = httpx.MockTransport(handler)
    cfg = make_cfg(tmp_path, transport=transport)

    run_rq4.run(cfg, gold=gold)  # must not raise

    rows = _read_rows(cfg.out_dir / "turns.jsonl")
    assert len(rows) == 30 * len(run_rq4.ARMS)
    failed = [r for r in rows if r["dispatch_failed"]]
    ok = [r for r in rows if not r["dispatch_failed"]]
    assert failed, "expected at least one dispatch_failed row"
    assert all(r["scored"] is False and r["correct"] is None for r in failed)
    assert ok, "expected most turns to succeed despite one poisoned question"


def test_resume_skips_completed_work(tmp_path, gold):
    transport = httpx.MockTransport(make_agent_handler())
    cfg = make_cfg(tmp_path, transport=transport)
    run_rq4.run(cfg, gold=gold)
    first_rows = _read_rows(cfg.out_dir / "turns.jsonl")
    assert len(first_rows) == 30 * len(run_rq4.ARMS)

    counter = {"sessions_opened": 0}
    base_handler = make_agent_handler()

    def counting_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/sessions":
            counter["sessions_opened"] += 1
        return base_handler(request)

    cfg2 = make_cfg(tmp_path, transport=httpx.MockTransport(counting_handler))
    run_rq4.run(cfg2, gold=gold)  # same out_dir/registry as cfg

    second_rows = _read_rows(cfg.out_dir / "turns.jsonl")
    assert len(second_rows) == len(first_rows)  # no duplicate rows appended
    assert counter["sessions_opened"] == 0      # nothing was re-dispatched


def test_run_one_turn_threads_call_records_to_reflection(gold):
    """A real session's spans must produce non-empty CallRecords for
    reflection to learn from — a turn scored only from `stats` (no Heimdall
    call shapes at all) would starve A2/A3's `aggregate()` of everything
    except pass/fail."""
    items = run_rq4.build_schedule(gold, seed=SEED, replication=1,
                                   hr_employee_id=HR_EMPLOYEE_ID)
    item = items[0]
    handler = make_agent_handler()
    client = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    agent = run_rq4.AgentClient(client=client, sleep=lambda s: None)

    row, calls = run_rq4.run_one_turn(
        agent=agent, phoenix=_SpanfulPhoenix(), item=item, arm="A2", replication=1,
        config_ref="agent_config_rq4_A2_i1@1", gold=gold, dry_run=False)

    assert row["dispatch_failed"] is False
    assert len(calls) == 1
    assert calls[0].tool == "mcp_query"
    assert calls[0].schema == "dm_core"


def test_dry_run_walks_the_schedule_without_a_transport(tmp_path, gold):
    cfg = run_rq4.RunConfig(
        seed=SEED, replications=1, epochs=1, concurrency=2,
        hr_employee_id=HR_EMPLOYEE_ID, run_id="dry", out_dir=tmp_path / "dry",
        dry_run=True, registry_db=str(tmp_path / "registry.db"),
    )
    run_rq4.run(cfg, gold=gold)
    rows = _read_rows(cfg.out_dir / "turns.jsonl")
    assert len(rows) == 30 * len(run_rq4.ARMS)
    assert all(r["dry_run"] for r in rows)
    assert all(not r["dispatch_failed"] for r in rows)
    assert any(r["scored"] for r in rows)


def test_reflection_commits_new_configs_after_an_epoch(tmp_path, gold):
    """Two epochs so a reflection cycle actually runs; asserts A2/A3 pick up
    a real (reflected) memory_ref for epoch 1 while A1 never does."""
    transport = httpx.MockTransport(make_agent_handler())
    cfg = make_cfg(tmp_path, transport=transport, epochs=2)
    run_rq4.run(cfg, gold=gold)

    registry = Registry(cfg.registry_db)
    try:
        for instance in (1, 2, 3):
            a1_cfg = registry.get_json(
                registry.head(run_rq4.config_name("A1", instance)).hash)
            assert a1_cfg["memory_strategy"] == "none"

            a2_cfg = registry.get_json(
                registry.head(run_rq4.config_name("A2", instance)).hash)
            assert a2_cfg["memory_strategy"] == "reflected"
            assert a2_cfg["memory_ref"].startswith(f"memory_A2_i{instance}@")

            a3_cfg = registry.get_json(
                registry.head(run_rq4.config_name("A3", instance)).hash)
            assert a3_cfg["memory_strategy"] == "reflected"
            assert a3_cfg["memory_ref"].startswith("memory_A3_fleet@")
    finally:
        registry.close()

    rows = _read_rows(cfg.out_dir / "turns.jsonl")
    assert len(rows) == 30 * len(run_rq4.ARMS) * 2  # two epochs


# --------------------------------------------------------------------- utils


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
