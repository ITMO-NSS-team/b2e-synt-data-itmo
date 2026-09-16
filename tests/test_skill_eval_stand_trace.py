"""Stand trace polling waits for the requested session, not any HTTP 200."""
from __future__ import annotations

from sim.skill_eval.stand import StandClient


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
