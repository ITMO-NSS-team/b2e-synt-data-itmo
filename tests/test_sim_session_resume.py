"""Session ownership and OpenCode recovery, with scripted CLI output (no API calls)."""
from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sim.agent.app import create_app
from sim.agent.config import AgentConfig
from sim.agent.opencode import API_PRICE_ENV, OpenCodeHarness
from sim.agent.progress import ProgressBoard
from sim.agent.store import Store


def _stream(message_id, native_id, tokens, *, answer="answer", cost=0.01):
    rows = [
        {"type": "step_start", "timestamp": 1000,
         "part": {"messageID": message_id}},
        {"type": "text", "part": {"messageID": message_id, "text": answer}},
        {"type": "step_finish", "timestamp": 1100,
         "part": {"messageID": message_id, "reason": "stop", "cost": cost,
                  "tokens": {"input": tokens, "output": 3, "reasoning": 2,
                             "cache": {"read": 7, "write": 0}}}},
    ]
    return "\n".join(json.dumps({**row, "sessionID": native_id}) for row in rows)


@pytest.fixture
def runtime(tmp_path, monkeypatch, fingerprint):
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "test-only")
    for variable in API_PRICE_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    scripts, calls = [], []

    def popen(argv, **kwargs):
        calls.append({"argv": argv, **kwargs})
        output, code = scripts.pop(0)
        return SimpleNamespace(stdout=io.StringIO(output), returncode=code,
                               wait=lambda: None, kill=lambda: None)

    monkeypatch.setattr("sim.agent.opencode.subprocess.Popen", popen)
    config = AgentConfig(harness="open_code", conversation_mode="resume")
    fingerprint = replace(fingerprint, harness="open_code")
    state = SimpleNamespace(
        store=Store(tmp_path / "agent.db"), progress=ProgressBoard(),
        harness_status="ready", config=config,
    )

    def restart():
        state.store.close()
        state.store = Store(tmp_path / "agent.db")
        state.progress = ProgressBoard()
        state.harness = OpenCodeHarness(
            heimdall_url="http://unused", heimdall_token="test",
            session_root=str(tmp_path / "sessions"),
            opencode_home=str(tmp_path / "opencode-home"),
        )
        return TestClient(create_app(state))

    state.build_fingerprint = lambda ref: (
        replace(fingerprint, harness=state.config.harness), state.config, "system")
    client = restart()
    sid = client.post("/sessions", json={"employee_id": "42"}).json()["session_id"]
    yield SimpleNamespace(state=state, client=client, sid=sid, scripts=scripts,
                          calls=calls, restart=restart,
                          url=f"/sessions/{sid}/messages")
    state.store.close()


def test_resume_after_restart_attributes_usage_to_each_turn(runtime, spans):
    r = runtime
    r.scripts.extend([(_stream("m1", "native-1", 10), 0),
                      (_stream("m2", "native-2", 20), 0),
                      (_stream("m3", "native-2", 30), 0)])
    first = r.client.post(r.url, json={"content": "first"}).json()
    r.client = r.restart()
    second = r.client.post(r.url, json={"content": "second"}).json()
    third = r.client.post(r.url, json={"content": "third"}).json()

    assert "--session" not in r.calls[0]["argv"]
    for call, expected in zip(r.calls[1:], ["native-1", "native-2"]):
        assert call["argv"][call["argv"].index("--session") + 1] == expected
        assert call["cwd"] == r.calls[0]["cwd"]
        assert call["env"]["XDG_DATA_HOME"] == r.calls[0]["env"]["XDG_DATA_HOME"]
    assert [turn["stats"]["prompt_tokens"] for turn in (first, second, third)] == [17, 27, 37]
    assert [turn["stats"]["iterations"] for turn in (first, second, third)] == [1, 1, 1]
    assert all(turn["stats"]["completion_tokens"] == 5 for turn in (first, second, third))
    session = r.state.store.get_session(r.sid)
    assert session["session_harness"] == "open_code"
    assert session["claude_session_id"] == "native-2"
    llm = [s for s in spans.get_finished_spans() if s.name == "llm.messages.create"]
    assert [s.attributes["b2e.llm.message_id"] for s in llm] == ["m1", "m2", "m3"]
    assert [s.attributes["llm.token_count.prompt"] for s in llm] == [17, 27, 37]
    assert len({s.context.trace_id for s in llm}) == 3
    stored = [m["stats"]["prompt_tokens"] for m in r.state.store.messages(r.sid)
              if m["role"] == "assistant"]
    assert stored == [17, 27, 37]


@pytest.mark.parametrize("fresh_fails", [False, True])
def test_missing_native_history_retries_once_and_reports_loss(runtime, spans, fresh_fails):
    r = runtime
    r.state.store.bind_harness_session(r.sid, "open_code", "missing")
    error = json.dumps({"type": "error", "error": {"message": "Session not found"}})
    r.scripts.extend([(error, 1), (error, 1) if fresh_fails else
                      (_stream("fresh", "new-native", 10), 0)])
    response = r.client.post(r.url, json={"content": "continue"})
    assert response.status_code == (502 if fresh_fails else 200)
    assert len(r.calls) == 2
    assert "--session" in r.calls[0]["argv"]
    assert "--session" not in r.calls[1]["argv"]
    assert "without history" in response.text
    assert "Session not found" in response.text
    root = next(s for s in spans.get_finished_spans() if s.name == "b2e.turn")
    assert root.attributes["b2e.resume_failed"] is True
    if not fresh_fails:
        assert r.state.store.get_session(r.sid)["claude_session_id"] == "new-native"
        assert response.json()["stats"]["iterations"] == 1
        assert response.json()["stats"]["cost_usd"] == 0.01


@pytest.mark.parametrize("failed_cost,fresh_cost,total_cost,cost_mode", [
    (0.01, 0.02, 0.03, "reported"),
    (0, 0.02, None, "unavailable"),
    (0.01, 0, None, "unavailable"),
    (0, 0, None, "subscription"),
])
def test_failed_resume_usage_is_not_discarded(
    runtime, spans, failed_cost, fresh_cost, total_cost, cost_mode,
):
    r = runtime
    r.state.store.bind_harness_session(r.sid, "open_code", "old")
    failed = _stream("failed", "old", 10, answer="", cost=failed_cost) + "\n" + json.dumps(
        {"type": "error", "error": {"message": "resume failed"}})
    r.scripts.extend([(failed, 1), (_stream("fresh", "new", 20, cost=fresh_cost), 0)])
    response = r.client.post(r.url, json={"content": "continue"})
    assert response.status_code == 200
    stats = response.json()["stats"]
    assert stats["prompt_tokens"] == 44
    assert stats["completion_tokens"] == 10
    assert stats["iterations"] == 2
    if total_cost is None:
        assert stats["cost_usd"] is None
    else:
        assert stats["cost_usd"] == pytest.approx(total_cost)
    assert stats["cost_mode"] == cost_mode
    llm = [s for s in spans.get_finished_spans() if s.name == "llm.messages.create"]
    assert len(llm) == 2
    assert sum(s.attributes["llm.token_count.prompt"] for s in llm) == 44


@pytest.mark.parametrize("bound,answer", [(False, ""), (True, "partial answer")])
def test_only_an_empty_failed_resume_is_retried(runtime, spans, bound, answer):
    r = runtime
    if bound:
        r.state.store.bind_harness_session(r.sid, "open_code", "old")
    output = _stream("failed", "old", 10, answer=answer) + "\n" + json.dumps(
        {"type": "error", "error": "failure"})
    r.scripts.append((output, 1))
    response = r.client.post(r.url, json={"content": "question"})
    assert response.status_code == (200 if answer else 502)
    assert len(r.calls) == 1
    assert "without history" not in response.text


@pytest.mark.parametrize("owner,target", [
    ("claude_code", "open_code"), ("open_code", "claude_code"),
    (None, "open_code"), (None, "claude_code"),
])
def test_incompatible_session_rejected_before_appending_or_invoking(runtime, owner, target):
    r = runtime
    r.state.config = replace(r.state.config, harness=target)
    if owner:
        r.state.store.bind_harness_session(r.sid, owner, "foreign")
    else:
        # Old deployments stored both harnesses here without recording ownership.
        r.state.store._conn.execute(
            "UPDATE sessions SET claude_session_id = 'legacy' WHERE id = ?", (r.sid,))
        r.state.store._conn.commit()
    response = r.client.post(r.url, json={"content": "continue"})
    assert response.status_code == 409
    assert "Create a new B2E session" in response.text
    assert r.calls == []
    assert r.state.store.messages(r.sid) == []


def test_stateless_turn_ignores_existing_binding_and_never_retries(runtime, spans):
    r = runtime
    r.state.config = replace(r.state.config, conversation_mode="stateless")
    r.state.store.bind_harness_session(r.sid, "claude_code", "foreign")
    r.scripts.append((json.dumps({"type": "error", "error": "failed"}), 1))
    assert r.client.post(r.url, json={"content": "question"}).status_code == 502
    assert len(r.calls) == 1
    assert "--session" not in r.calls[0]["argv"]
    assert r.state.store.get_session(r.sid)["claude_session_id"] == "foreign"


def test_legacy_binding_migration_preserves_id_without_guessing_owner(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, employee_id TEXT, "
                   "config_ref TEXT, created_at REAL, fingerprint TEXT, metadata TEXT, "
                   "claude_session_id TEXT)")
        db.execute("INSERT INTO sessions VALUES "
                   "('old', '42', 'cfg@1', 0, '{}', '{}', 'native')")
    store = Store(path)
    try:
        assert store.get_session("old")["claude_session_id"] == "native"
        assert store.get_session("old")["session_harness"] is None
        store.bind_claude_session("old", "new")
        assert store.get_session("old")["session_harness"] == "claude_code"
    finally:
        store.close()
