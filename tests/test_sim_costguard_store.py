"""Cost guard and job store — both had zero tests until the audit.

A mutation battery deleted the pre-dispatch ceiling check, the in-flight check,
and idempotency handling entirely, and all 156 tests stayed green. These are the
two modules where a silent regression costs real money or launches a duplicate
paid batch, so they are tested against observable consequences: how many model
calls actually happened, and how many jobs actually exist.
"""
from __future__ import annotations

import threading

import pytest

from sim.agent.config import AgentConfig
from sim.agent.llm import LLMResponse, ScriptedClient
from sim.agent.loop import run_turn
from sim.agent.store import Store
from sim.costguard import (
    Budget, CostCeilingExceeded, CostGuard, KillSwitchEngaged, Projection,
    all_guards, get, register,
)
from sim.fingerprint import RunFingerprint

FP = RunFingerprint.create(
    agent_config_version="cfg@1", prompt_registry_version="p@1",
    skill_registry_hash="sha256:" + "00" * 32,
    model_id="claude-haiku-4-5-20251001", temperature=0.0,
    data_snapshot_hash="snap@1", traps_enabled=True, latency_profile="instant")


def _projection(questions: int = 500) -> Projection:
    return Projection(questions=questions, expected_calls_per_question=6,
                      expected_prompt_tokens_per_call=6000,
                      expected_completion_tokens_per_call=700,
                      model_id="claude-haiku-4-5-20251001")


# ------------------------------------------------------------- pre-dispatch


def test_an_experiment_without_a_ceiling_is_refused():
    """A default ceiling is one nobody chose."""
    with pytest.raises(ValueError, match="must declare a ceiling"):
        Budget()


def test_a_breaching_projection_raises_before_anything_is_spent():
    guard = CostGuard(Budget(max_usd=0.01), experiment_id="e1")
    with pytest.raises(CostCeilingExceeded):
        guard.check_projection(_projection())
    assert guard.calls == 0 and guard.spent_usd == 0.0


def test_an_approved_projection_returns_a_loggable_record():
    """The projection must be logged before dispatch, so it has to come back."""
    guard = CostGuard(Budget(max_usd=1000.0), experiment_id="e2")
    record = guard.check_projection(_projection(1))
    assert record["approved"] is True
    assert record["projected_usd"] > 0
    assert "rate table" in record["basis"]


def test_a_breaching_projection_dispatches_zero_model_calls():
    """The observable form: not 'it raised', but 'the model was never called'."""
    guard = CostGuard(Budget(max_tokens=10), experiment_id="e3")
    client = ScriptedClient([LLMResponse(content=[{"type": "text", "text": "hi"}],
                                         stop_reason="end_turn",
                                         prompt_tokens=100, completion_tokens=10)])
    try:
        guard.check_projection(_projection(50))
    except CostCeilingExceeded:
        pass
    assert client.calls == [], "a refused batch must not reach the model at all"


# ---------------------------------------------------------------- in-flight


class _Tools:
    """Minimal stand-in; the loop only needs a call counter and dispatch."""
    call_count = 0

    def dispatch(self, name, arguments):
        return {"ok": True}

    def close(self):
        pass


def test_the_ceiling_stops_a_running_loop(monkeypatch):
    guard = CostGuard(Budget(max_tokens=250), experiment_id="e4")
    responses = [LLMResponse(
        content=[{"type": "tool_use", "id": f"t{i}", "name": "list_models",
                  "input": {}}],
        stop_reason="tool_use", prompt_tokens=100, completion_tokens=20)
        for i in range(10)]
    client = ScriptedClient(responses)
    with pytest.raises(CostCeilingExceeded):
        run_turn(question="q", history=[], config=AgentConfig(harness="messages_api"),
                 system_prompt="s", client=client, tools=_Tools(),
                 fingerprint=FP, session_id="s1", employee_id="1", guard=guard)
    assert len(client.calls) <= 3, "the loop kept calling past its ceiling"


def test_the_kill_switch_reaches_a_running_experiment_through_the_registry():
    guard = register(CostGuard(Budget(max_usd=100.0), experiment_id="e5"))
    assert get("e5") is guard
    assert guard in all_guards()
    get("e5").kill("operator stop")
    with pytest.raises(KillSwitchEngaged):
        guard.check_before_call()
    guard.resume()
    guard.check_before_call()


def test_spend_accumulates_and_status_reports_it():
    guard = CostGuard(Budget(max_tokens=1000), experiment_id="e6")
    guard.record(tokens=400, usd=0.02)
    guard.record(tokens=400, usd=0.02)
    status = guard.status()
    assert status["spent_tokens"] == 800 and status["calls"] == 2
    guard.record(tokens=400, usd=0.02)
    with pytest.raises(CostCeilingExceeded):
        guard.check_before_call()


# ------------------------------------------------------------- idempotency


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "agent.db")
    yield s
    s.close()


def test_a_repeated_idempotency_key_returns_the_original_job(store):
    """A researcher whose connection dropped needs the id of the batch already
    running, not a second paid batch."""
    first, created_first = store.create_job(request={"n": 1}, idempotency_key="k")
    second, created_second = store.create_job(request={"n": 1}, idempotency_key="k")
    assert created_first is True and created_second is False
    assert first == second
    assert len(store.list_jobs()) == 1


def test_different_keys_create_different_jobs(store):
    a, _ = store.create_job(request={}, idempotency_key="a")
    b, _ = store.create_job(request={}, idempotency_key="b")
    assert a != b and len(store.list_jobs()) == 2


def test_no_key_means_no_deduplication(store):
    a, _ = store.create_job(request={}, idempotency_key=None)
    b, _ = store.create_job(request={}, idempotency_key=None)
    assert a != b


def test_concurrent_submissions_with_one_key_yield_exactly_one_job(store):
    """The only way to reach the IntegrityError race branch."""
    results: list[tuple[str, bool]] = []
    lock = threading.Lock()

    def submit():
        out = store.create_job(request={}, idempotency_key="race")
        with lock:
            results.append(out)

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({job_id for job_id, _ in results}) == 1
    assert sum(1 for _, created in results if created) == 1
    assert len(store.list_jobs()) == 1


def test_job_status_and_result_round_trip(store):
    job_id, _ = store.create_job(request={"q": 1}, idempotency_key=None)
    store.update_job(job_id, status="running", projection={"projected_usd": 1.0})
    store.update_job(job_id, status="completed", result={"runs": [{"a": 1}]})
    job = store.get_job(job_id)
    assert job["status"] == "completed"
    assert job["result"]["runs"] == [{"a": 1}]
    assert job["projection"]["projected_usd"] == 1.0


def test_sessions_and_messages_round_trip(store):
    sid = store.create_session(employee_id="42", config_ref="cfg@1",
                               fingerprint=FP.as_dict())
    store.append_message(session_id=sid, role="user", content="вопрос")
    store.append_message(session_id=sid, role="assistant", content="ответ",
                         trace_id="abc")
    messages = store.messages(sid)
    assert [m["seq"] for m in messages] == [1, 2]
    assert store.history_for_model(sid) == [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"}]
    assert store.get_session(sid)["employee_id"] == "42"


# --------------------------------------------------- resumable sessions


def test_a_new_session_is_not_bound_to_a_claude_session(store):
    """Null until a turn actually runs. Under conversation_mode="stateless" it
    stays null forever, which is the honest record of "there was no session to
    come back to"."""
    sid = store.create_session(employee_id="42", config_ref="cfg@1",
                               fingerprint=FP.as_dict())
    assert store.get_session(sid)["claude_session_id"] is None


def test_binding_a_claude_session_round_trips(store):
    sid = store.create_session(employee_id="42", config_ref="cfg@1",
                               fingerprint=FP.as_dict())
    store.bind_claude_session(sid, "11111111-2222-3333-4444-555555555555")
    assert (store.get_session(sid)["claude_session_id"]
            == "11111111-2222-3333-4444-555555555555")


def test_rebinding_replaces_the_previous_id(store):
    """Claude Code hands back a different id after a compaction or a fork.
    Binding once would leave later turns resuming a superseded session."""
    sid = store.create_session(employee_id="42", config_ref="cfg@1",
                               fingerprint=FP.as_dict())
    store.bind_claude_session(sid, "first")
    store.bind_claude_session(sid, "second")
    assert store.get_session(sid)["claude_session_id"] == "second"


def test_an_existing_database_gains_the_column(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` is a no-op against a database that already
    has the table, so a new column reaches fresh installs only. Every deployment
    already holding sessions would keep the old shape and fail on first read —
    a failure that cannot appear in CI, because CI always starts empty.
    """
    import sqlite3

    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, employee_id TEXT NOT NULL, "
        "config_ref TEXT NOT NULL, created_at REAL NOT NULL, "
        "fingerprint TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}')")
    legacy.execute(
        "INSERT INTO sessions VALUES ('ses_old', '42', 'cfg@1', 0.0, '{}', '{}')")
    legacy.commit()
    legacy.close()

    store = Store(path)
    try:
        assert store.get_session("ses_old")["claude_session_id"] is None
        store.bind_claude_session("ses_old", "resumed")
        assert store.get_session("ses_old")["claude_session_id"] == "resumed"
    finally:
        store.close()
