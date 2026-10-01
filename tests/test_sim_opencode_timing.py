"""Native timing attribution without provider requests or synthetic TTFT."""
import json
import urllib.error
import urllib.request

import pytest

from sim import telemetry
from sim.agent.claude_code import emit_spans, parse_stream as parse_claude
from sim.agent.config import AgentConfig
from sim.agent.opencode import OpenCodeHarness, parse_stream
from sim.agent.opencode_timing import NativeTimings, SOURCE, collect_native_timings
from sim.benchmark.execution import AgentTurn, trace_observations


def native_span(identity="one", *, start=1000, end=1800, first=100, finish=600, session="s"):
    values = {"session.id": session, "ai.response.msToFirstChunk": first,
              "ai.response.msToFinish": finish}
    return {"name": "ai.streamText.doStream", "traceId": "trace", "spanId": identity,
            "startTimeUnixNano": str(start * 1_000_000), "endTimeUnixNano": str(end * 1_000_000),
            "attributes": [{"key": key, "value": {"stringValue" if isinstance(value, str)
                                                   else "doubleValue": value}}
                           for key, value in values.items() if value is not None]}


def result():
    events = []
    for mid, start, end in (("m1", 1110, 1810), ("m2", 2110, 2810)):
        events.extend([
            {"type": "step_start", "timestamp": start, "sessionID": "s", "part": {"messageID": mid}},
            {"type": "step_finish", "timestamp": end, "sessionID": "s",
             "part": {"messageID": mid, "tokens": {}}},
        ])
    return parse_stream("\n".join(map(json.dumps, events)), model="test")


def capture(*spans):
    timing = NativeTimings()
    timing.ingest({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]})
    return timing


def test_native_timings_reach_existing_spans_and_benchmark(spans, fingerprint):
    timing = capture(native_span(), native_span(),  # duplicate OTLP batch
                     native_span("two", start=2000, end=2800, first=50, finish=400),
                     native_span("history", start=100, end=400),
                     native_span("background", start=3000, end=3400),
                     native_span("other", session="other"))
    turn = result()
    timing.apply(turn, started_ns=900_000_000)
    assert turn.api_duration_ms == 1000  # SDK finish latency, not span/tool duration
    assert turn.ttft_stream_ms == 100
    assert turn.time_to_request_ms == 100
    assert turn.ttft_ms is None  # first chunk is not necessarily a text token
    assert turn.timing_calls == 2
    assert [call.api_duration_ms for call in turn.llm_calls] == [600, 400]
    with telemetry.start_run("b2e.turn", fingerprint=fingerprint,
                             session_id="s", employee_id="e") as root:
        emit_spans(turn, root=root, harness_name="open_code")
    recorded = spans.get_finished_spans()
    llm = [s for s in recorded if s.name == "llm.messages.create"]
    assert len(llm) == 2
    assert llm[0].attributes["b2e.llm.ttft_stream_ms"] == 100
    assert llm[0].attributes["b2e.llm.timing_source"] == SOURCE
    root = next(s for s in recorded if s.name == "b2e.turn")
    observed = trace_observations(AgentTurn(answer="", trace={"spans": [
        {"name": "b2e.turn", "attributes": dict(root.attributes)}]}))
    assert observed["api_duration_ms"] == 1000
    assert observed["ttft_stream_ms"] == 100
    assert observed["time_to_request_ms"] == 100
    assert observed["ttft_ms"] is None


@pytest.mark.parametrize("first,finish", [(0, 0), (None, None), (-1, -1), (float("nan"), float("inf"))])
def test_zero_is_valid_but_missing_or_invalid_timings_are_unknown(first, finish):
    turn = result()
    turn.llm_calls = turn.llm_calls[:1]
    capture(native_span(first=first, finish=finish)).apply(turn, started_ns=900_000_000)
    expected = 0 if first == 0 else None
    assert turn.ttft_stream_ms == expected
    assert turn.api_duration_ms == expected


def test_partial_or_ambiguous_capture_does_not_publish_complete_turn_timings():
    for timing in (capture(), capture(native_span()),
                   capture(native_span(), native_span("ambiguous"))):
        turn = result()
        timing.apply(turn, started_ns=900_000_000)
        assert turn.api_duration_ms is None
        assert turn.ttft_stream_ms is None
        assert turn.time_to_request_ms is None


def test_claude_reported_zero_timings_are_exported(spans, fingerprint):
    turn = parse_claude(json.dumps({"type": "result", "result": "ok",
                                   "duration_api_ms": 0, "ttft_ms": 0}))
    with telemetry.start_run("b2e.turn", fingerprint=fingerprint,
                             session_id="s", employee_id="e") as root:
        emit_spans(turn, root=root)
    root = next(s for s in spans.get_finished_spans() if s.name == "b2e.turn")
    assert root.attributes["b2e.turn.api_duration_ms"] == 0
    assert root.attributes["b2e.turn.ttft_ms"] == 0
    assert "b2e.turn.ttft_stream_ms" not in root.attributes


def test_private_receiver_accepts_native_batches_and_discards_content():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    span = native_span()
    span["attributes"].append({"key": "ai.prompt", "value": {"stringValue": "sensitive prompt"}})
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}
    with collect_native_timings() as timing:
        assert timing.endpoint.startswith("http://127.0.0.1:")
        request = urllib.request.Request(timing.endpoint + "/v1/traces",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=2) as response:
            assert response.status == 200
        request = urllib.request.Request(timing.endpoint + "/wrong-path", data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(request, timeout=2)
        assert error.value.code == 404
        assert len(timing.requests) == 1
        assert "sensitive" not in json.dumps(timing.requests)
    with pytest.raises(urllib.error.URLError):
        opener.open(timing.endpoint + "/v1/traces", timeout=2)


@pytest.mark.parametrize("first_duration,expected", [(600, 1000), (None, None)])
def test_resume_recovery_keeps_first_request_and_sums_only_complete_api_timings(
    tmp_path, monkeypatch, first_duration, expected,
):
    runner = OpenCodeHarness(heimdall_url="http://unused", heimdall_token="test",
                             workdir=str(tmp_path))
    failed, fresh = result(), result()
    failed.is_error, failed.error = True, "history missing"
    failed.api_duration_ms, fresh.api_duration_ms = first_duration, 400
    failed.ttft_stream_ms, fresh.ttft_stream_ms = 100, 50
    failed.time_to_request_ms, fresh.time_to_request_ms = 25, 10
    failed.timing_calls = fresh.timing_calls = 2
    attempts = iter((failed, fresh))
    monkeypatch.setattr(runner, "_invoke", lambda *args, **kwargs: next(attempts))
    turn = runner.run(question="hello", config=AgentConfig(conversation_mode="resume"),
                       system_prompt="system", employee_id="e", resume_session_id="old")
    assert turn.resumed_failed
    assert turn.api_duration_ms == expected
    assert turn.ttft_stream_ms == 100
    assert turn.time_to_request_ms == 25
    assert turn.timing_calls == 4
