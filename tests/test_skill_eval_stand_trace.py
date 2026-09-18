"""Stand trace polling waits for the requested session, not any HTTP 200."""
from __future__ import annotations

from sim.skill_eval.stand import StandClient
from sim.skill_eval.types import EvalCase, SessionSpec


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
