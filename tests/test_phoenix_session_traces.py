"""A session lookup must never return spans belonging to another run."""
from __future__ import annotations

import httpx
import pytest

from sim.research.phoenix_client import PhoenixClient, PhoenixUnavailable


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


@pytest.mark.parametrize("limit, match", [(1, "span limit"), (20000, "cursor repeated")])
def test_trace_rejects_truncation_or_repeated_cursor(limit, match):
    def handler(request):
        return httpx.Response(200, json={
            "data": [_span("ses-target", "trace-target", "root")],
            "next_cursor": "same",
        })
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://phoenix")
    client = PhoenixClient("http://phoenix", client=http)
    with pytest.raises(PhoenixUnavailable, match=match):
        client.spans_for_trace("trace-target", limit=limit)


def test_latest_traces_respects_span_budget_and_marks_incomplete_data():
    requested_limits = []

    def handler(request):
        if request.url.path.endswith("/traces"):
            return httpx.Response(200, json={"data": [{"trace_id": "new"}, {"trace_id": "old"}]})
        requested_limits.append(int(request.url.params["limit"]))
        first = not request.url.params.get("cursor")
        spans = [_span(None, "new", str(i)) for i in ([1, 2] if first else [3])]
        return httpx.Response(200, json={"data": spans, "next_cursor": "next" if first else "more"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://phoenix")
    client = PhoenixClient("http://phoenix", client=http)
    traces = client.latest_traces(span_cap=3)
    assert requested_limits == [3, 1]
    assert len(traces[0]["spans"]) == 3
    assert traces[1]["spans"] == []
    assert all(t["span_limit_reached"] for t in traces)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("include", [False, True])
def test_live_span_loading_can_omit_reconstructed_inputs_without_losing_outputs(single, include):
    def handler(request):
        if request.url.path.endswith("/traces"):
            return httpx.Response(200, json={"data": [{"trace_id": "trace-target"}]})
        span = _span("session", "trace-target", "root")
        span["attributes"].update({
            "llm.input_messages.0.message.content": "long conversation",
            "llm": {"input_messages": ["long conversation"], "output_messages": ["answer"]},
            "input.value": "question", "output.value": "answer",
        })
        return httpx.Response(200, json={"data": [span]})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://phoenix")
    client = PhoenixClient("http://phoenix", client=http)
    spans = client.spans_for_trace("trace-target", include_input_messages=include) if single else client.latest_traces(include_input_messages=include)[0]["spans"]
    attrs = spans[0]["attributes"]
    assert ("llm.input_messages.0.message.content" in attrs) is include
    assert ("input_messages" in attrs["llm"]) is include
    assert attrs["llm"]["output_messages"] == ["answer"]
    assert attrs["input.value"] == "question"
    assert attrs["output.value"] == "answer"


def test_page_size_budget_is_reported_and_single_trace_never_returns_partial_data():
    def handler(request):
        if request.url.path.endswith("/traces"):
            return httpx.Response(200, json={"data": [{"trace_id": "trace-target"}]})
        span = _span("session", "trace-target", "root")
        span["attributes"]["output.value"] = "x" * 1000
        return httpx.Response(200, json={"data": [span]})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://phoenix")
    client = PhoenixClient("http://phoenix", client=http)
    result = client.latest_traces(max_bytes=10)
    assert result[0]["span_limit_reached"] is True
    assert result[0]["load_limit_reason"] == "page_size"
    assert result[0]["spans"] == []
    with pytest.raises(PhoenixUnavailable, match="page size budget"):
        client.spans_for_trace("trace-target", max_bytes=10)
