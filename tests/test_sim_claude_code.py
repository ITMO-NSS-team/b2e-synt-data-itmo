"""Claude Code harness: tool policy and stream parsing.

These are the parts testable without spending money or needing the CLI. The
parts that are actually the boundary — whether the permission matcher refuses
`runner x; python3 -c …` — cannot be unit-tested, because they are enforced by
Claude Code itself. Those were probed against a live CLI and the results are
recorded in docs/skill-execution-threat-model.md §10; re-run that suite on every
CLI upgrade, because the matcher's behaviour is not a published contract.
"""
from __future__ import annotations

import json

import pytest

from sim.agent.claude_code import (
    DENIED_TOOLS, HEIMDALL_TOOLS, MCP_SERVER_NAME, TOOL_LOADER,
    ClaudeCodeHarness, parse_stream,
)
from sim.agent.config import AgentConfig


@pytest.fixture()
def harness():
    return ClaudeCodeHarness(
        heimdall_url="http://heimdall-emulator:8081",
        heimdall_token="tok",
        bridge_path="/app/heimdall/bridge.py",
        runner_path="/opt/skills/run",
    )


# ------------------------------------------------------------- tool policy


def test_file_readers_are_denied():
    """The corpus is a directory of files and truth/people.json is the answer
    key. A file tool is a path straight to it, bypassing the API entirely."""
    for tool in ("Read", "Glob", "Grep", "NotebookRead"):
        assert tool in DENIED_TOOLS, tool


def test_authoring_and_egress_tools_are_denied():
    for tool in ("Write", "Edit", "MultiEdit", "NotebookEdit", "Task",
                 "WebFetch", "WebSearch"):
        assert tool in DENIED_TOOLS, tool


def test_allowed_tools_are_exactly_heimdall_plus_runner_plus_loader(harness):
    allowed = harness.allowed_tools(AgentConfig())
    bash = [t for t in allowed if t.startswith("Bash")]
    assert bash == ["Bash(/opt/skills/run:*)"], bash
    assert TOOL_LOADER in allowed
    mcp = [t for t in allowed if t.startswith("mcp__")]
    assert mcp and all(t.startswith(f"mcp__{MCP_SERVER_NAME}__") for t in mcp)


def test_bash_is_never_granted_unrestricted(harness):
    """A bare `Bash` entry would grant every command, silently."""
    assert "Bash" not in harness.allowed_tools(AgentConfig())


def test_tool_subset_narrows_the_mcp_surface(harness):
    """The subset is an experimental variable; a wildcard would ignore it."""
    narrow = AgentConfig(tool_subset=("mcp_query",))
    mcp = [t for t in harness.allowed_tools(narrow) if t.startswith("mcp__")]
    assert f"mcp__{MCP_SERVER_NAME}__mcp_query" in mcp
    assert f"mcp__{MCP_SERVER_NAME}__describe_model" not in mcp


def test_denied_and_allowed_never_overlap(harness):
    allowed = {t.split("(")[0] for t in harness.allowed_tools(AgentConfig())}
    assert not (allowed & set(DENIED_TOOLS))


# ----------------------------------------------------------------- argv


def test_argv_pins_reproducibility_flags(harness):
    argv = harness.build_argv("q", config=AgentConfig(), system_suffix="s",
                              mcp_config_path="/tmp/mcp.json")
    joined = " ".join(argv)
    # No host settings, hooks or plugins: a run whose behaviour depends on
    # ~/.claude is not reproducible and the fingerprint cannot describe it.
    assert "--setting-sources" in argv
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert "--no-session-persistence" in argv
    assert "--output-format" in joined and "stream-json" in joined


def test_argv_carries_the_configured_model(harness):
    config = AgentConfig(model_id="claude-haiku-4-5-20251001")
    argv = harness.build_argv("q", config=config, system_suffix="s",
                              mcp_config_path="/tmp/mcp.json")
    assert argv[argv.index("--model") + 1] == "claude-haiku-4-5-20251001"


# ------------------------------------------------------------ mcp config


def test_mcp_config_carries_the_acting_identity(harness):
    cfg = harness.mcp_config("2457060")
    env = cfg["mcpServers"][MCP_SERVER_NAME]["env"]
    assert env["HEIMDALL_EMPLOYEE_ID"] == "2457060"
    assert env["HEIMDALL_URL"] == "http://heimdall-emulator:8081"


def test_two_employees_get_different_mcp_configs(harness):
    a = harness.mcp_config("111")["mcpServers"][MCP_SERVER_NAME]["env"]
    b = harness.mcp_config("222")["mcpServers"][MCP_SERVER_NAME]["env"]
    assert a["HEIMDALL_EMPLOYEE_ID"] != b["HEIMDALL_EMPLOYEE_ID"]


def test_child_env_is_an_allowlist_not_a_denylist(harness, monkeypatch):
    """Built by allowlist so a newly-added parent secret does not leak by
    default — a deny-list is wrong until someone remembers to update it."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-TESTONLY")
    monkeypatch.setenv("POSTGRES_PASSWORD", "should-not-propagate")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "should-not-propagate")
    env = harness.child_env()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-TESTONLY"
    assert "POSTGRES_PASSWORD" not in env
    assert "TELEGRAM_BOT_TOKEN" not in env


def test_child_env_refuses_without_a_credential(harness, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no model credential"):
        harness.child_env()


# ------------------------------------------------------------ stream parse


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


RESULT_EVENT = {
    "type": "result", "is_error": False, "num_turns": 3,
    "result": "Три витрины: A, B, C.",
    "session_id": "abc-123", "total_cost_usd": 0.0489, "duration_ms": 16100,
    "permission_denials": [],
    "usage": {"input_tokens": 30, "output_tokens": 1068,
              "cache_read_input_tokens": 24807, "cache_creation_input_tokens": 12},
    "modelUsage": {"claude-haiku-4-5-20251001":
                   {"contextWindow": 200000, "maxOutputTokens": 32000}},
}


def test_parse_extracts_answer_and_usage():
    out = parse_stream(_stream(RESULT_EVENT))
    assert out.answer.startswith("Три витрины")
    assert (out.input_tokens, out.output_tokens) == (30, 1068)
    assert out.cache_read_tokens == 24807
    assert out.cost_usd == pytest.approx(0.0489)
    assert out.session_id == "abc-123"


def test_parse_reads_limits_from_the_session_not_a_constant():
    """Claude Code reports 32 000 max output for Haiku 4.5, not the 64 000 the
    original spec assumed. Hardcoding either would be a silent lie."""
    out = parse_stream(_stream(RESULT_EVENT))
    assert out.context_window == 200000
    assert out.max_output_tokens == 32000


def test_parse_counts_heimdall_calls_separately_from_skill_runs():
    stream = _stream(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "1", "name": "ToolSearch", "input": {}}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "2",
             "name": f"mcp__{MCP_SERVER_NAME}__mcp_query", "input": {}}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "3", "name": "Bash", "input": {}}]}},
        RESULT_EVENT)
    out = parse_stream(stream)
    assert len(out.tool_calls) == 3
    assert out.heimdall_calls == 1
    assert out.skill_runs == 1


def test_permission_denials_surface_attempted_tools():
    """The measurable form of the premise: not "did it behave" but "did it try"."""
    event = {**RESULT_EVENT, "permission_denials": [
        {"tool_name": "Bash"}, {"tool_name": "Bash"}, {"tool_name": "Write"}]}
    out = parse_stream(_stream(event))
    assert len(out.permission_denials) == 3
    assert out.attempted_forbidden_tools == ["Bash", "Write"]


def test_parse_survives_garbage_lines():
    out = parse_stream("not json\n" + json.dumps(RESULT_EVENT) + "\n\n")
    assert out.answer.startswith("Три витрины")


def test_parse_reports_an_errored_session():
    out = parse_stream(_stream({**RESULT_EVENT, "is_error": True,
                                "api_error_status": "429"}))
    assert out.is_error and out.error == "429"


def test_empty_stream_does_not_crash():
    out = parse_stream("")
    assert out.answer == "" and out.total_tokens == 0
