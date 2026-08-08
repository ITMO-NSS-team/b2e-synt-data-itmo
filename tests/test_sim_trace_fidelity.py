"""What a trace has to contain before anyone can measure anything from it.

The corpus recorded through 2026-08-05 could not answer two questions it was
built to answer: where a turn spends its time, and how many tokens a turn costs.
Both had the same cause — the Claude Code harness runs the model out of process,
and the spans were reconstructed after the subprocess exited, so every TOOL span
had zero duration and no token count reached Phoenix at all.

These tests pin the parts that are fixable in process: real tool timing taken
live off the event stream, and the token counts the session already reported.
"""
from __future__ import annotations

import json
import threading
import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sim import telemetry
from sim.agent.claude_code import (
    MCP_SERVER_NAME,
    ToolSpanRecorder,
    emit_spans,
    parse_stream,
)
from sim.fingerprint import RunFingerprint

QUERY_TOOL = f"mcp__{MCP_SERVER_NAME}__mcp_query"


# ------------------------------------------------------------------ fixtures


def _root(fingerprint: RunFingerprint):
    return telemetry.start_run(
        "b2e.turn", fingerprint=fingerprint, session_id="ses_test",
        employee_id="1599763", question="сколько сотрудников?")


def _tool_use(call_id: str, name: str = QUERY_TOOL) -> dict:
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": call_id, "name": name,
         "input": {"schema": "dm_core", "logic_model": "employee_actual"}}]}}


def _tool_result(call_id: str, text: str = '{"data": []}') -> dict:
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": call_id, "content": text}]}}


RESULT_EVENT = {
    "type": "result", "is_error": False, "num_turns": 3,
    "result": "В компании 294 118 сотрудников.",
    "session_id": "abc-123", "total_cost_usd": 0.0489, "duration_ms": 16100,
    "permission_denials": [],
    "usage": {"input_tokens": 30, "output_tokens": 1068,
              "cache_read_input_tokens": 24807, "cache_creation_input_tokens": 12},
    "modelUsage": {"claude-haiku-4-5-20251001":
                   {"contextWindow": 200000, "maxOutputTokens": 32000}},
}


def _by_name(exporter: InMemorySpanExporter, name: str) -> list:
    return [s for s in exporter.get_finished_spans() if s.name == name]


# ------------------------------------------------------- token accounting


def test_prompt_tokens_include_the_cached_prefix():
    """`usage.input_tokens` is the *uncached* remainder. Reporting it alone as
    the prompt size told the corpus a 24 807-token prompt was 30 tokens."""
    out = parse_stream("\n".join(json.dumps(e) for e in [RESULT_EVENT]))
    assert out.prompt_tokens == 30 + 24807 + 12


def test_total_tokens_counts_what_the_model_actually_read():
    out = parse_stream("\n".join(json.dumps(e) for e in [RESULT_EVENT]))
    assert out.total_tokens == 30 + 24807 + 12 + 1068


def test_root_span_carries_the_token_counts_the_session_reported(spans, fingerprint):
    """Phoenix reads `llm.token_count.*`. Without them its token and cost columns
    are NULL for every turn the default harness ever ran."""
    out = parse_stream("\n".join(json.dumps(e) for e in [RESULT_EVENT]))
    with _root(fingerprint) as root:
        emit_spans(out, root=root)

    attrs = _by_name(spans, "b2e.turn")[0].attributes
    assert attrs["llm.token_count.prompt"] == 30 + 24807 + 12
    assert attrs["llm.token_count.completion"] == 1068
    assert attrs["llm.token_count.total"] == 30 + 24807 + 12 + 1068


def test_root_span_separates_cache_reads_from_fresh_prompt(spans, fingerprint):
    """A turn that reads 24 807 cached tokens and a turn that pays for 24 807
    fresh ones cost different money and must not look identical."""
    out = parse_stream("\n".join(json.dumps(e) for e in [RESULT_EVENT]))
    with _root(fingerprint) as root:
        emit_spans(out, root=root)

    attrs = _by_name(spans, "b2e.turn")[0].attributes
    assert attrs["llm.token_count.prompt_details.cache_read"] == 24807
    assert attrs["llm.token_count.prompt_details.cache_write"] == 12


# ------------------------------------------------------------ live timing


def test_tool_span_records_the_time_the_tool_actually_took(spans, fingerprint):
    """The whole point. Spans replayed after the subprocess exits are 0.000 s."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1"))
        time.sleep(0.05)
        recorder.observe(_tool_result("call-1"))
        recorder.finish()

    tool_spans = _by_name(spans, f"tool.{QUERY_TOOL}")
    assert len(tool_spans) == 1
    elapsed = (tool_spans[0].end_time - tool_spans[0].start_time) / 1e9
    assert elapsed >= 0.05


def test_tool_span_keeps_the_input_and_what_came_back(spans, fingerprint):
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1"))
        recorder.observe(_tool_result("call-1", '{"data": [{"person_id": "x"}]}'))
        recorder.finish()

    attrs = _by_name(spans, f"tool.{QUERY_TOOL}")[0].attributes
    assert attrs["tool.name"] == QUERY_TOOL
    assert "employee_actual" in attrs["input.value"]
    assert "person_id" in attrs["output.value"]


def test_tool_spans_nest_under_the_turn_even_from_the_reader_thread(spans, fingerprint):
    """`on_event` is called from the thread draining the CLI's stdout. OTel
    context is thread-local, so a span opened there lands in its own trace
    unless the root's context is carried across explicitly."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        root_context = root.get_span_context()

        def drain() -> None:
            recorder.observe(_tool_use("call-1"))
            recorder.observe(_tool_result("call-1"))

        reader = threading.Thread(target=drain)
        reader.start()
        reader.join()
        recorder.finish()

    tool_span = _by_name(spans, f"tool.{QUERY_TOOL}")[0]
    assert tool_span.context.trace_id == root_context.trace_id
    assert tool_span.parent.span_id == root_context.span_id


def test_a_tool_that_never_returned_still_leaves_a_span(spans, fingerprint):
    """A turn killed by the watchdog mid-tool is exactly when the trace matters
    most; dropping the open span hides the call that was hanging."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1"))
        recorder.finish()

    tool_spans = _by_name(spans, f"tool.{QUERY_TOOL}")
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["b2e.tool.unfinished"] is True


def test_live_spans_are_not_duplicated_by_the_closing_replay(spans, fingerprint):
    """`emit_spans` still has to work for turns with no live recorder; it must
    not emit a second copy of what the recorder already recorded."""
    stream = "\n".join(json.dumps(e) for e in
                       [_tool_use("call-1"), _tool_result("call-1"), RESULT_EVENT])
    out = parse_stream(stream)

    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1"))
        recorder.observe(_tool_result("call-1"))
        recorder.finish()
        emit_spans(out, root=root, recorder=recorder)

    assert len(_by_name(spans, f"tool.{QUERY_TOOL}")) == 1


def test_without_a_recorder_the_replay_still_produces_tool_spans(spans, fingerprint):
    stream = "\n".join(json.dumps(e) for e in
                       [_tool_use("call-1"), _tool_result("call-1"), RESULT_EVENT])
    out = parse_stream(stream)

    with _root(fingerprint) as root:
        emit_spans(out, root=root)

    assert len(_by_name(spans, f"tool.{QUERY_TOOL}")) == 1


def test_root_span_reports_tool_time_so_model_time_is_derivable(spans, fingerprint):
    """Turn duration minus tool time is the model's share. Neither number was
    available before, so an 81-second average could not be attributed to
    anything."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1"))
        time.sleep(0.05)
        recorder.observe(_tool_result("call-1"))
        recorder.finish()
        emit_spans(parse_stream(json.dumps(RESULT_EVENT)), root=root,
                   recorder=recorder)

    attrs = _by_name(spans, "b2e.turn")[0].attributes
    assert attrs["b2e.turn.tool_time_ms"] >= 50


# ------------------------------------------------------ the wired-up turn


class _FakeHarness:
    """Replays a scripted event stream through whatever `run` is handed."""

    def __init__(self, events: list[dict]) -> None:
        self.events = events
        self.calls: list[dict] = []

    def run(self, **kwargs) -> object:
        self.calls.append(kwargs)
        on_event = kwargs.get("on_event")
        for event in self.events:
            if on_event is not None:
                on_event(event)
            if event is not self.events[-1]:
                time.sleep(0.01)
        return parse_stream("\n".join(json.dumps(e) for e in self.events))


class _FakeStore:
    def bind_claude_session(self, *args, **kwargs) -> None:
        return None


def _run_turn(harness, fingerprint, metadata=None):
    from types import SimpleNamespace

    from sim.agent.app import _run_claude_code
    from sim.agent.config import AgentConfig
    from sim.agent.progress import ProgressBoard

    state = SimpleNamespace(harness=harness, harness_status="ok",
                            progress=ProgressBoard(), store=_FakeStore())
    return _run_claude_code(
        state, {"employee_id": "1599763"}, AgentConfig(),
        "system prompt", "сколько сотрудников?", fingerprint, "ses_test",
        metadata or {})


def test_turn_result_reports_the_prompt_the_model_actually_read(spans, fingerprint):
    """`stats.prompt_tokens` in agent.db is what the research API and the
    Telegram footer both quote. It was reporting the uncached remainder."""
    harness = _FakeHarness([RESULT_EVENT])
    result = _run_turn(harness, fingerprint)
    assert result.prompt_tokens == 30 + 24807 + 12


def test_an_experiment_turn_keeps_its_raw_stream(spans, fingerprint):
    """Instrumentation changes; a stream kept on disk can be re-read against the
    new one. A turn discarded at source cannot be re-derived by anybody."""
    harness = _FakeHarness([RESULT_EVENT])
    _run_turn(harness, fingerprint, metadata={"experiment_id": "exp_1"})
    assert harness.calls[0]["keep_stream"] is True


def test_an_ordinary_turn_does_not_keep_its_raw_stream(spans, fingerprint):
    """A Telegram conversation is not a measurement, and its stream carries the
    whole transcript. Keeping every one of them is a disk leak with personal
    data in it."""
    harness = _FakeHarness([RESULT_EVENT])
    _run_turn(harness, fingerprint)
    assert harness.calls[0]["keep_stream"] is False


def test_the_cost_guard_counts_the_cached_prefix_it_paid_for(spans, fingerprint):
    """A token ceiling that ignores cache reads is off by ~25x on this stack, so
    an experiment could burn its real budget while the guard reported it had
    barely started."""
    from sim import costguard
    from sim.costguard import Budget, CostGuard

    guard = costguard.register(CostGuard(Budget(max_tokens=10_000_000),
                                         experiment_id="exp_tokens"))
    _run_turn(_FakeHarness([RESULT_EVENT]), fingerprint,
              metadata={"experiment_id": "exp_tokens"})
    assert guard.spent_tokens == 30 + 24807 + 12 + 1068


def test_a_wired_turn_produces_one_timed_tool_span(spans, fingerprint):
    """End to end: the recorder is actually connected to the harness, the spans
    nest under the turn, and the closing replay does not duplicate them."""
    harness = _FakeHarness([_tool_use("call-1"), _tool_result("call-1"), RESULT_EVENT])
    _run_turn(harness, fingerprint)

    tool_spans = _by_name(spans, f"tool.{QUERY_TOOL}")
    assert len(tool_spans) == 1
    assert (tool_spans[0].end_time - tool_spans[0].start_time) / 1e9 >= 0.01

    root = _by_name(spans, "b2e.turn")[0]
    assert tool_spans[0].parent.span_id == root.context.span_id
    assert root.attributes["b2e.turn.tool_time_ms"] >= 10


def test_recorder_ignores_events_that_are_not_tool_traffic(spans, fingerprint):
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe({"type": "system", "subtype": "init"})
        recorder.observe({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "думаю"}]}})
        recorder.finish()

    assert not [s for s in spans.get_finished_spans() if s.name.startswith("tool.")]
