"""A session lookup must never return spans belonging to another run."""
from __future__ import annotations

import httpx

from sim.research.phoenix_client import PhoenixClient


def _span(session_id: str | None, trace_id: str, span_id: str) -> dict:
    attributes = {"session.id": session_id} if session_id else {}
    return {
        "name": "b2e.turn" if session_id else "iteration.1",
        "context": {"trace_id": trace_id, "span_id": span_id},
        "attributes": attributes,
    }


def test_spans_for_session_resolves_roots_then_fetches_only_their_trace() -> None:
    requests: list[httpx.Request] = []
    target_root = _span("ses-target", "trace-target", "root")
    target_child = _span(None, "trace-target", "child")
    foreign = _span("ses-foreign", "trace-foreign", "foreign")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("attribute"):
            return httpx.Response(200, json={"data": [target_root]})
        return httpx.Response(200, json={
            # Defensive filtering must discard this even if Phoenix ever
            # ignores trace_id in the same way it ignored the old filter.
            "data": [target_root, target_child, foreign],
        })

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://phoenix")
    client = PhoenixClient("http://phoenix", project="project", client=http)

    spans = client.spans_for_session("ses-target")

    assert {span["context"]["trace_id"] for span in spans} == {"trace-target"}
    assert requests[0].url.params.get("attribute") == "session.id:ses-target"
    assert requests[0].url.params.get("filter") is None
    assert requests[1].url.params.get_list("trace_id") == ["trace-target"]


def test_spans_for_session_does_not_fetch_project_when_root_is_absent() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://phoenix")
    client = PhoenixClient("http://phoenix", project="project", client=http)

    assert client.spans_for_session("ses-missing") == []
    assert len(requests) == 1
