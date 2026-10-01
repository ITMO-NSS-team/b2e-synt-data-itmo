"""CLI denials, backend errors, shell attempts and actual runner executions."""
import json

import pytest

from sim import telemetry
from sim.agent.claude_code import ToolSpanRecorder, emit_spans, parse_stream as parse_claude
from sim.agent.opencode import _observe_open_code, parse_stream
from sim.benchmark.execution import AgentTurn, trace_observations

DENIED = ("The user has specified a rule which prevents you from using this specific "
          "tool call. Here are some of the relevant rules []")
CLAUDE_DENIED = "Permission to use Bash has been denied because Claude Code is running in don't ask mode."
RUNNER = "/custom/runner"
SKILL = {"ok": True, "skill": {"name": "count", "hash": "a" * 64, "state": "active"},
         "wall_ms": 1, "result": 3}


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("name,command,output,error,denied,runs,successes", [
    ("bash", "grep x file", DENIED, True, True, 0, 0),
    ("bash", "grep x file", CLAUDE_DENIED, True, True, 0, 0),
    ("bash", "grep x file", {"name": "PermissionRejectedError"}, True, True, 0, 0),
    ("heimdall_mcp_query", "", {"code": "forbidden", "status": 403}, True, False, 0, 0),
    ("bash", "false", "command exited 1", True, False, 0, 0),
    ("bash", "pwd", "/app", False, False, 0, 0),
    ("bash", f"{RUNNER} hash", SKILL, False, False, 1, 1),
    ("bash", f"{RUNNER} hash", {**SKILL, "ok": False, "error": {"kind": "timeout"}},
     False, False, 1, 0),
    ("bash", f"{RUNNER} hash", {"ok": False, "error": {"kind": "not-executable"}},
     False, False, 0, 0),
    ("bash", f"echo {RUNNER}", SKILL, False, False, 0, 0),
])
def test_outcomes_match_in_live_and_replayed_spans(
    spans, fingerprint, live, name, command, output, error, denied, runs, successes,
):
    raw_output = json.dumps(output) if isinstance(output, dict) and "name" not in output else output
    row = {"type": "tool_use", "timestamp": 1200, "part": {
        "messageID": "m1", "callID": "call1", "tool": name,
        "state": {"status": "error" if error else "completed", "input": {"command": command},
                  "error" if error else "output": raw_output,
                  "time": {"start": 1000, "end": 1200}},
    }}
    stream = json.dumps(row)
    result = parse_stream(stream, model="zai-coding-plan/glm")
    result.runner_path = RUNNER
    assert len(result.permission_denials) == int(denied)
    assert result.attempted_forbidden_tools == (["Bash"] if denied else [])
    assert result.bash_attempts == int(name == "bash")
    assert result.skill_runs == runs
    assert result.skill_successes == successes
    with telemetry.start_run("b2e.turn", fingerprint=fingerprint,
                             session_id="s", employee_id="42") as root:
        recorder = ToolSpanRecorder(root, runner_path=RUNNER) if live else None
        if recorder:
            _observe_open_code(stream, recorder.observe)
            recorder.finish()
        emit_spans(result, root=root, recorder=recorder, harness_name="open_code")
    recorded = spans.get_finished_spans()
    tool = next(s for s in recorded if s.name.startswith("tool."))
    expected_error = error or (command.startswith(RUNNER) and isinstance(output, dict)
                               and output.get("ok") is False)
    assert tool.attributes["b2e.tool.permission_denied"] is denied
    assert tool.attributes["b2e.tool.is_error"] == expected_error
    assert tool.status.status_code.name == ("ERROR" if expected_error else "OK")
    assert tool.attributes["b2e.tool.skill_executed"] == bool(runs)
    trace = {"spans": [{"name": s.name, "attributes": dict(s.attributes),
                         "status_code": s.status.status_code.name} for s in recorded]}
    observed = trace_observations(AgentTurn("answer", trace=trace))
    assert observed["permission_denials"] == int(denied)
    assert observed["failed_tool_calls"] == int(expected_error)
    assert observed["skill_runs"] == runs
    assert observed["skill_successes"] == successes


def test_legacy_denials_override_false_zero_without_conflating_http_403():
    trace = {"spans": [
        {"attributes": {"b2e.turn.tool_calls": 3, "b2e.permission_denials": 0,
                        "b2e.turn.skill_runs": 2}},
        *[{"name": "tool.Bash", "status_code": "UNSET",
           "attributes": {"tool.name": "Bash", "output.value": message}}
          for message in [DENIED, CLAUDE_DENIED]],
        {"name": "heimdall.mcp_query", "attributes": {
            "b2e.heimdall.endpoint": "mcp_query", "b2e.http.status": 403}},
    ]}
    observed = trace_observations(AgentTurn("answer", trace=trace))
    assert observed["permission_denials"] == 2
    assert observed["http_error_count"] == 1
    assert observed["failed_tool_calls"] == 3
    assert observed["bash_attempts"] == 2
    assert observed["skill_runs"] is None  # Legacy counter is not trustworthy.


def test_claude_structured_result_and_envelope_denials_are_not_double_counted():
    rows = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "c", "name": "Bash", "input": {"command": "pwd"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "c", "is_error": True,
             "content": [{"type": "text", "text": CLAUDE_DENIED}]}]}},
        {"type": "result", "permission_denials": [{"tool_use_id": "c", "tool_name": "Bash"}]},
    ]
    result = parse_claude("\n".join(map(json.dumps, rows)))
    assert len(result.permission_denials) == 1
    assert result.tool_calls[0]["permission_denied"] is True
    assert result.tool_calls[0]["is_error"] is True
    assert result.skill_runs == 0


def test_open_code_running_and_completed_updates_count_once():
    part = {"tool": "bash", "callID": "c", "messageID": "m"}
    rows = [{"type": "tool_use", "part": {**part, "state": {"status": status}}}
            for status in ("running", "completed")]
    result = parse_stream("\n".join(map(json.dumps, rows)), model="model")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["status"] == "completed"
    events = []
    _observe_open_code(json.dumps(rows[0]), events.append)
    assert not events
