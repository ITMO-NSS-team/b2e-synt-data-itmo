"""Spans for the work that happens outside this process.

Stage 1 (`test_sim_trace_fidelity.py`) times tool calls from the CLI's event
stream — the boundaries of each call as seen from outside. This file covers the
two layers underneath, both of which run in subprocesses this stack owns:

* the HTTP call the MCP bridge makes to Heimdall, and
* the skill execution the sandbox worker performs.

Neither was visible in any trace recorded before 2026-08-08. See
`docs/subprocess-tracing-plan.md`.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sim.agent.claude_code import (
    MCP_SERVER_NAME,
    ClaudeCodeHarness,
    ToolSpanRecorder,
    emit_spans,
    parse_stream,
)
from sim.agent.config import AgentConfig
from sim.fingerprint import RunFingerprint
from sim import telemetry

BRIDGE = pathlib.Path(__file__).resolve().parents[1] / "heimdall" / "bridge.py"
QUERY_TOOL = f"mcp__{MCP_SERVER_NAME}__mcp_query"


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def harness(tmp_path) -> ClaudeCodeHarness:
    return ClaudeCodeHarness(
        heimdall_url="http://127.0.0.1:1", heimdall_token="t",
        bridge_path=str(BRIDGE), runner_path="/opt/skills/run",
        session_root=str(tmp_path))


def _root(fingerprint: RunFingerprint):
    return telemetry.start_run(
        "b2e.turn", fingerprint=fingerprint, session_id="ses_test",
        employee_id="1599763", question="сколько сотрудников?")


def _by_name(exporter: InMemorySpanExporter, name: str) -> list:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _tool_use(call_id: str, name: str, payload: dict | None = None) -> dict:
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": call_id, "name": name,
         "input": payload or {"schema": "dm_core",
                              "logic_model": "employee_actual"}}]}}


def _tool_result(call_id: str, text: str) -> dict:
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": call_id, "content": text}]}}


# ------------------------------------- stage 2: the bridge's own trace log


def _run_bridge(tmp_path, tool: str = "mcp_query") -> list[dict]:
    """Drive the real bridge as a subprocess and read back what it logged.

    Points at a dead port on purpose: the interest is in what the log records
    about a call, and a connection refusal is a perfectly good call to record.
    """
    log = tmp_path / "heimdall.jsonl"
    env = {**os.environ,
           "HEIMDALL_URL": "http://127.0.0.1:1",
           "HR_TRACE_LOG": str(log),
           "TRACEPARENT": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": tool, "arguments": {}}})
    proc = subprocess.run([sys.executable, str(BRIDGE)], input=request + "\n",
                          capture_output=True, text=True, env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr[:400]
    return [json.loads(line) for line in log.read_text("utf-8").splitlines() if line]


def test_the_bridge_records_when_a_call_started_not_only_how_long_it_took(tmp_path):
    """A duration with no anchor cannot be placed on a timeline, which is the
    whole reason the span exists."""
    before = time.time()
    records = _run_bridge(tmp_path)
    assert len(records) == 1
    assert before <= records[0]["ts_start"] <= time.time()


def test_the_bridge_records_the_trace_it_was_called_under(tmp_path):
    records = _run_bridge(tmp_path)
    assert records[0]["traceparent"].startswith(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-")


def test_the_bridge_records_the_http_call_it_actually_made(tmp_path):
    records = _run_bridge(tmp_path)
    assert (records[0]["method"], records[0]["path"]) == ("POST", "/api/v1/mcp/query/")
    assert records[0]["status"] == 503          # nothing is listening on :1


def test_the_bridge_stays_silent_when_no_log_is_configured(tmp_path):
    """Its default is to write nothing: it also runs outside this stack."""
    env = {**os.environ, "HEIMDALL_URL": "http://127.0.0.1:1"}
    env.pop("HR_TRACE_LOG", None)
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "mcp_query", "arguments": {}}})
    proc = subprocess.run([sys.executable, str(BRIDGE)], input=request + "\n",
                          capture_output=True, text=True, env=env, timeout=30)
    assert proc.returncode == 0
    assert not list(tmp_path.iterdir())


def test_the_harness_gives_the_bridge_a_log_and_the_current_trace(spans, harness):
    """Without this the bridge's log is dead code — which is what it was.

    Takes `spans` for the tracer provider: with no provider registered there is
    no span context to encode, and the harness correctly emits no traceparent.
    """
    with telemetry.get_tracer().start_as_current_span("outer"):
        env = harness.mcp_config("1599763", AgentConfig(),
                                 trace_log="/tmp/turn.jsonl")[
            "mcpServers"][MCP_SERVER_NAME]["env"]
    assert env["HR_TRACE_LOG"] == "/tmp/turn.jsonl"
    assert env["TRACEPARENT"].startswith("00-")


def test_the_harness_omits_the_log_when_nobody_asked_for_one(harness):
    env = harness.mcp_config("1599763", AgentConfig())[
        "mcpServers"][MCP_SERVER_NAME]["env"]
    assert "HR_TRACE_LOG" not in env


# ------------------------------------- stage 2: bridge records become spans


def _bridge_record(ts_start: float, duration_ms: float = 40.0, **over) -> dict:
    record = {"tool": "mcp_query", "args_keys": ["logic_model", "schema"],
              "status": 200, "duration_ms": duration_ms, "code": None,
              "rows": 3, "bytes": 512, "ts_start": ts_start,
              "method": "POST", "path": "/api/v1/mcp/query/",
              "traceparent": ""}
    record.update(over)
    return record


def test_a_bridge_record_becomes_a_span_under_the_tool_that_caused_it(
        spans, fingerprint):
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", QUERY_TOOL))
        started = time.time()
        time.sleep(0.02)
        recorder.observe(_tool_result("call-1", '{"data": []}'))
        recorder.finish()

        result = parse_stream("")
        result.bridge_calls = [_bridge_record(started)]
        emit_spans(result, root=root, recorder=recorder)

    http = _by_name(spans, "heimdall.mcp_query")
    assert len(http) == 1
    tool_span = _by_name(spans, f"tool.{QUERY_TOOL}")[0]
    assert http[0].parent.span_id == tool_span.context.span_id


def test_the_heimdall_span_carries_the_status_and_shape_of_the_response(
        spans, fingerprint):
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", QUERY_TOOL))
        started = time.time()
        recorder.observe(_tool_result("call-1", '{"data": []}'))
        recorder.finish()

        result = parse_stream("")
        result.bridge_calls = [_bridge_record(started, rows=42)]
        emit_spans(result, root=root, recorder=recorder)

    attrs = _by_name(spans, "heimdall.mcp_query")[0].attributes
    assert attrs["b2e.http.status"] == 200
    assert attrs["b2e.http.method"] == "POST"
    assert attrs["b2e.http.path"] == "/api/v1/mcp/query/"
    assert attrs["b2e.heimdall.rows"] == 42


def test_the_heimdall_span_lasts_as_long_as_the_bridge_measured(spans, fingerprint):
    """The duration comes from the bridge, which is inside the call. Nothing
    here re-measures it, and nothing here may invent it."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", QUERY_TOOL))
        started = time.time()
        recorder.observe(_tool_result("call-1", '{"data": []}'))
        recorder.finish()

        result = parse_stream("")
        result.bridge_calls = [_bridge_record(started, duration_ms=137.5)]
        emit_spans(result, root=root, recorder=recorder)

    span = _by_name(spans, "heimdall.mcp_query")[0]
    assert (span.end_time - span.start_time) / 1e6 == pytest.approx(137.5, abs=1.0)


def test_a_failed_call_marks_the_span_and_keeps_the_error_code(spans, fingerprint):
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", QUERY_TOOL))
        started = time.time()
        recorder.observe(_tool_result("call-1", "{}"))
        recorder.finish()

        result = parse_stream("")
        result.bridge_calls = [_bridge_record(started, status=400,
                                              code="invalid-argument")]
        emit_spans(result, root=root, recorder=recorder)

    span = _by_name(spans, "heimdall.mcp_query")[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["b2e.heimdall.error_code"] == "invalid-argument"


def test_a_record_matching_no_tool_call_is_kept_under_the_turn_and_flagged(
        spans, fingerprint):
    """Dropping it would hide a real HTTP call. Silently reparenting it without
    saying so would be worse — the ratio it feeds would look verified."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.finish()
        result = parse_stream("")
        result.bridge_calls = [_bridge_record(time.time())]
        emit_spans(result, root=root, recorder=recorder)

    span = _by_name(spans, "heimdall.mcp_query")[0]
    root_span = _by_name(spans, "b2e.turn")[0]
    assert span.parent.span_id == root_span.context.span_id
    assert span.attributes["b2e.trace.correlation"] == "unmatched"


def test_two_calls_to_one_tool_land_under_their_own_tool_spans(spans, fingerprint):
    """Paging a mart is the same tool several times over. Collapsing them would
    destroy exactly the "calls per answer" number this exists to measure."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", QUERY_TOOL))
        first = time.time()
        time.sleep(0.02)
        recorder.observe(_tool_result("call-1", "{}"))
        recorder.observe(_tool_use("call-2", QUERY_TOOL))
        second = time.time()
        time.sleep(0.02)
        recorder.observe(_tool_result("call-2", "{}"))
        recorder.finish()

        result = parse_stream("")
        result.bridge_calls = [_bridge_record(first, duration_ms=1),
                               _bridge_record(second, duration_ms=1)]
        emit_spans(result, root=root, recorder=recorder)

    http = _by_name(spans, "heimdall.mcp_query")
    assert len({s.parent.span_id for s in http}) == 2


def test_a_corrupt_bridge_log_line_does_not_cost_the_turn_its_trace(
        spans, fingerprint):
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.finish()
        result = parse_stream("")
        result.bridge_calls = [{"nonsense": True}, _bridge_record(time.time())]
        emit_spans(result, root=root, recorder=recorder)

    assert _by_name(spans, "b2e.turn")
    assert len(_by_name(spans, "heimdall.mcp_query")) == 1


# --------------------------------------------- stage 3: the skill sandbox


RUNNER_OUTPUT = json.dumps({
    "ok": True,
    "result": {"headcount": 294118},
    "wall_ms": 12.4,
    "peak_rss_kb": 20480,
    "rejected_imports": ["socket"],
    "skill": {"name": "headcount_by_dimension", "hash": "sha256:" + "a" * 64,
              "state": "active"},
})


def test_an_approved_skill_run_gets_its_own_span_under_the_bash_call(
        spans, fingerprint):
    """Before this, an approved skill execution appeared in the trace as
    `tool.Bash` and nothing else."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", "Bash", {"command": "/opt/skills/run x"}))
        recorder.observe(_tool_result("call-1", RUNNER_OUTPUT))
        recorder.finish()

    sandbox = _by_name(spans, "sandbox.execute")
    assert len(sandbox) == 1
    assert sandbox[0].parent.span_id == _by_name(spans, "tool.Bash")[0].context.span_id


def test_the_sandbox_span_proves_which_bytes_ran(spans, fingerprint):
    """`b2e.skill.hash` is the digest the runner re-verified before dispatch.
    Keeping it out of the trace defeated the reason it is recorded at all."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", "Bash", {"command": "/opt/skills/run x"}))
        recorder.observe(_tool_result("call-1", RUNNER_OUTPUT))
        recorder.finish()

    attrs = _by_name(spans, "sandbox.execute")[0].attributes
    assert attrs["b2e.skill.hash"] == "sha256:" + "a" * 64
    assert attrs["b2e.skill.name"] == "headcount_by_dimension"
    assert attrs["b2e.skill.state"] == "active"
    assert attrs["b2e.sandbox.wall_ms"] == pytest.approx(12.4)
    assert attrs["b2e.sandbox.peak_rss_kb"] == 20480
    assert "socket" in attrs["b2e.sandbox.rejected_imports"]


def test_a_refused_skill_is_recorded_as_a_refusal_not_a_success(spans, fingerprint):
    refusal = json.dumps({"ok": False, "error": {
        "kind": "not-executable", "detail": "skill is not active"}})
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", "Bash", {"command": "/opt/skills/run x"}))
        recorder.observe(_tool_result("call-1", refusal))
        recorder.finish()

    attrs = _by_name(spans, "sandbox.execute")[0].attributes
    assert attrs["b2e.sandbox.exit_status"] == "not-executable"


def test_a_bash_call_that_is_not_a_skill_run_gets_no_sandbox_span(spans, fingerprint):
    """`echo` and `pwd` are permitted regardless of the allowlist. Manufacturing
    a sandbox span for them would invent executions that never happened."""
    with _root(fingerprint) as root:
        recorder = ToolSpanRecorder(root)
        recorder.observe(_tool_use("call-1", "Bash", {"command": "pwd"}))
        recorder.observe(_tool_result("call-1", "/app"))
        recorder.finish()

    assert not _by_name(spans, "sandbox.execute")
