"""Stand trace polling waits for the requested session, not any HTTP 200."""
from __future__ import annotations

from sim.skill_eval.stand import StandClient
from sim.skill_eval.types import EvalCase, SessionSpec
import httpx
import pytest


class Response:
    status_code = 200

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    def json(self) -> dict:
        return {
            # The outer id is echoed by research-api and is insufficient proof.
            "session_id": "ses-target",
            "tree": [{
                "name": "b2e.turn",
                "attributes": {"session.id": self._session_id},
                "context": {"trace_id": "trace-target"},
            }],
        }


class Client:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.calls = 0

    def get(self, _url: str) -> Response:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_wait_trace_ignores_foreign_200_until_target_arrives(monkeypatch) -> None:
    stand = object.__new__(StandClient)
    stand.trace_attempts = 2
    stand._client = Client([Response("ses-foreign"), Response("ses-target")])
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _seconds: None)

    trace = stand._wait_trace("ses-target")

    assert trace is not None
    assert stand._client.calls == 2


def test_wait_trace_returns_none_when_every_200_is_foreign(monkeypatch) -> None:
    stand = object.__new__(StandClient)
    stand.trace_attempts = 2
    stand._client = Client([Response("ses-a"), Response("ses-b")])
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _seconds: None)

    assert stand._wait_trace("ses-target") is None


class HttpResponse:
    def __init__(
        self, status_code: int, *, text: str = "", json_body: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self._json = json_body

    def json(self) -> dict:
        return self._json or {}


class ScriptedClient:
    def __init__(
        self, *, posts: list[HttpResponse], gets: list[Response] | None = None,
    ) -> None:
        self._posts = list(posts)
        self._gets = list(gets or [])
        self.get_calls = 0

    def post(self, _url: str, json: dict | None = None) -> HttpResponse:
        del json
        return self._posts.pop(0)

    def get(self, _url: str) -> Response:
        self.get_calls += 1
        return self._gets.pop(0)


def _case() -> EvalCase:
    return EvalCase(
        case_id="case-1", category="answerable", question="q",
        runtime_actor_employee_id="1", expected_skill=None,
        expected_skill_kind=None, gold={}, snapshot_id=None,
        business_task={}, raw={},
    )


def _spec() -> SessionSpec:
    return SessionSpec(employee_id="1", config_ref="agent@1", metadata={})


def test_http_error_keeps_the_full_body(monkeypatch) -> None:
    long_body = "traceback " + ("x" * 500)
    stand = object.__new__(StandClient)
    stand.trace_attempts = 1
    stand._client = ScriptedClient(posts=[HttpResponse(502, text=long_body)])
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _seconds: None)

    turn = stand.run(_case(), _spec())

    assert long_body in (turn.error or "")
    assert turn.trace is None
    assert stand._client.get_calls == 0


def test_message_http_error_still_stores_the_session_trace(monkeypatch) -> None:
    long_body = "agent failed " + ("y" * 500)
    stand = object.__new__(StandClient)
    stand.trace_attempts = 1
    stand._client = ScriptedClient(
        posts=[
            HttpResponse(200, text="{}", json_body={
                "session_id": "ses-target",
                "fingerprint": {"data_snapshot_hash": "snap@1"},
            }),
            HttpResponse(500, text=long_body),
        ],
        gets=[Response("ses-target")],
    )
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _seconds: None)

    turn = stand.run(_case(), _spec())

    assert long_body in (turn.error or "")
    assert turn.session_id == "ses-target"
    assert turn.trace is not None
    assert turn.trace_id == "trace-target"
    assert turn.live_snapshot_id == "snap@1"


def test_tool_free_arm_does_not_wait_for_denied_heimdall_attempts(monkeypatch) -> None:
    stand = object.__new__(StandClient)
    stand._client = ScriptedClient(posts=[
        HttpResponse(200, text="{}", json_body={
            "session_id": "ses-target",
            "fingerprint": {"data_snapshot_hash": "snap@1"},
        }),
        HttpResponse(200, text="{}", json_body={
            "answer": '{"result": null, "message": "no data access"}',
            "trace_id": "trace-target",
            "stats": {"heimdall_calls": 4},
            "errors": ["denied:mcp__heimdall__find_skills"],
        }),
    ])
    observed = []

    def wait_trace(session_id, *, trace_id=None, expected_heimdall_calls=0):
        observed.append((session_id, trace_id, expected_heimdall_calls))
        return {"spans": []}

    monkeypatch.setattr(stand, "_wait_trace", wait_trace)

    turn = stand.run(
        _case(),
        SessionSpec(
            employee_id="1", config_ref="agent@1",
            metadata={"heimdall_access": "disabled"},
        ),
    )

    assert observed == [("ses-target", "trace-target", 0)]
    assert turn.heimdall_calls == 4


def test_mixed_research_response_discards_foreign_spans():
    from sim.skill_eval.stand import _owned_trace
    mixed = {"tree": [
        {"name": "b2e.turn", "context": {"trace_id": "ours"},
         "attributes": {"session.id": "ses-ours"}, "children": [
             {"name": "heimdall.mcp_query", "context": {"trace_id": "ours"}},
         ]},
        {"name": "b2e.turn", "context": {"trace_id": "foreign"},
         "attributes": {"session.id": "ses-other"}, "children": [
             {"name": "heimdall.get_skill", "context": {"trace_id": "foreign"}},
         ]},
    ]}
    owned = _owned_trace(mixed, "ses-ours", "ours")
    assert len(owned["spans"]) == 2
    assert {s["context"]["trace_id"] for s in owned["spans"]} == {"ours"}
    foreign = _owned_trace(mixed, "ses-ours", "foreign")
    assert {s["context"]["trace_id"] for s in foreign["spans"]} == {"foreign"}


def test_direct_phoenix_reads_paginated_trace_waits_and_filters(monkeypatch):
    requests = []
    root = {"name": "b2e.turn", "context": {"trace_id": "ours", "span_id": "root"},
            "end_time": "2026-01-01T00:00:01Z", "attributes": {"session.id": "ses-ours"}}
    child = {"name": "heimdall.mcp_query", "context": {"trace_id": "ours", "span_id": "child"}}
    foreign = {"name": "heimdall.get_skill", "context": {"trace_id": "foreign", "span_id": "bad"}}
    def handler(request):
        requests.append(request)
        assert request.url.path == "/phoenix/v1/projects/b2e-itmo/spans"
        assert request.url.params["trace_id"] == "ours"
        assert "filter" not in request.url.params
        if request.url.params.get("cursor") == "page2":
            return httpx.Response(200, json={"data": [child, foreign]})
        return httpx.Response(200, json={"data": [root], "next_cursor": "page2"})
    stand = object.__new__(StandClient)
    stand.trace_backend = "phoenix"
    stand.trace_attempts = 3
    stand._phoenix_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://stand/phoenix")
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _: None)
    trace = stand._wait_trace("ses-ours", trace_id="ours")
    assert len(requests) == 4  # two complete identical reads
    assert len(trace["spans"]) == 2
    from sim.benchmark.execution import AgentTurn, trace_observations
    observed = trace_observations(AgentTurn("", trace=trace))
    assert observed["mcp_query_calls"] == 1
    assert observed["loaded_skills"] == []


def test_explicit_remote_url_wins_over_env_and_host_defaults_to_remote(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCHER_PASSWORD", raising=False)
    env = tmp_path / ".env"
    env.write_text("RESEARCHER_PASSWORD=test-only\nPUBLIC_URL=https://localhost:8443\n")
    stand = StandClient(env_file=str(env), public_url="https://remote:8443")
    try:
        assert stand.public_url == "https://remote:8443"
        assert stand._client.headers["Host"] == "remote"
        assert stand.phoenix_project == "b2e-itmo"
    finally:
        stand._client.close()
        stand.phoenix_http().close()


def test_direct_phoenix_waits_for_reported_heimdall_calls(monkeypatch):
    root = {"name": "b2e.turn", "context": {"trace_id": "ours", "span_id": "zzz-root"},
            "end_time": "2026-01-01", "attributes": {"session.id": "ses-ours"}}
    child = {"name": "heimdall.mcp_query", "context": {"trace_id": "ours", "span_id": "aaa-child"},
             "attributes": {"b2e.heimdall.endpoint": "mcp_query"}}
    calls = []
    def handler(request):
        calls.append(request)
        # A stable ended root alone is not proof that tool spans arrived.
        return httpx.Response(200, json={"data": [root] if len(calls) <= 2 else [root, child]})
    stand = object.__new__(StandClient)
    stand.trace_backend = "phoenix"
    stand.trace_attempts = 4
    stand._phoenix_http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://stand/phoenix")
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _: None)
    trace = stand._wait_trace("ses-ours", trace_id="ours", expected_heimdall_calls=1)
    assert len(calls) == 4
    from sim.skill_eval.scoring.trace import root_span_id
    assert root_span_id(trace) == "zzz-root"


def test_direct_phoenix_keeps_polling_while_spans_still_arrive(monkeypatch):
    root = {"name": "b2e.turn", "context": {"trace_id": "ours", "span_id": "root"},
            "end_time": "2026-01-01", "attributes": {"session.id": "ses-ours"}}
    calls = []

    def handler(request):
        calls.append(request)
        spans = [root]
        # More polls than trace_attempts: the old wall-clock loop would stop here.
        n_children = min(len(calls), 6)
        for index in range(n_children):
            spans.append({
                "name": "heimdall.mcp_query",
                "context": {"trace_id": "ours", "span_id": f"child-{index}"},
                "attributes": {"b2e.heimdall.endpoint": "mcp_query"},
            })
        return httpx.Response(200, json={"data": spans})

    stand = object.__new__(StandClient)
    stand.trace_backend = "phoenix"
    stand.trace_attempts = 2
    stand._phoenix_http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://stand/phoenix")
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _: None)
    trace = stand._wait_trace("ses-ours", trace_id="ours", expected_heimdall_calls=6)
    assert len(calls) == 7
    assert len(trace["spans"]) == 7


def test_direct_phoenix_keeps_polling_before_any_span_lands(monkeypatch):
    root = {"name": "b2e.turn", "context": {"trace_id": "ours", "span_id": "root"},
            "end_time": "2026-01-01", "attributes": {"session.id": "ses-ours"}}
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= 5:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"data": [root]})

    stand = object.__new__(StandClient)
    stand.trace_backend = "phoenix"
    stand.trace_attempts = 2
    stand._phoenix_http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://stand/phoenix")
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _: None)
    trace = stand._wait_trace("ses-ours", trace_id="ours")
    assert len(calls) == 7
    assert trace["spans"] == [root]


def test_stand_supports_internal_agent_and_phoenix_without_basic_auth(tmp_path):
    stand = StandClient(
        env_file=str(tmp_path / "missing.env"),
        public_url="http://b2e-agent:8082",
        agent_prefix="",
        phoenix_url="http://phoenix:6006",
        require_auth=False,
    )
    try:
        assert stand.agent_prefix == ""
        assert str(stand._client.base_url) == "http://b2e-agent:8082"
        assert str(stand.phoenix_http().base_url) == "http://phoenix:6006"
    finally:
        stand._client.close()
        stand.phoenix_http().close()


def test_stand_reads_phoenix_project_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCHER_PASSWORD", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "RESEARCHER_PASSWORD=test-only\nPHOENIX_PROJECT=custom-project\n",
        encoding="utf-8",
    )
    stand = StandClient(env_file=str(env), public_url="https://stand")
    try:
        assert stand.phoenix_project == "custom-project"
    finally:
        stand._client.close()
        stand.phoenix_http().close()


def test_stand_password_from_environment_does_not_need_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHER_PASSWORD", "from-env")
    missing = tmp_path / "missing.env"
    stand = StandClient(env_file=str(missing), public_url="https://stand")
    try:
        assert isinstance(stand._client.auth, httpx.BasicAuth)
    finally:
        stand._client.close()
        stand.phoenix_http().close()


def test_failed_phoenix_never_falls_back_to_unfiltered_research(monkeypatch):
    stand = object.__new__(StandClient)
    stand.trace_backend = "phoenix"
    stand.trace_timeout = 2
    stand._phoenix_http = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(503)), base_url="https://stand/phoenix")
    clock = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr("sim.skill_eval.stand.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("sim.skill_eval.stand.time.sleep", lambda _: None)
    assert stand._wait_trace("ses-ours", trace_id="ours") is None
