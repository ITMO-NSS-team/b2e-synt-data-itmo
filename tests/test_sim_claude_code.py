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
import os
import pathlib
import re
import subprocess
import sys

import pytest

from sim.agent.claude_code import (
    DENIED_TOOLS, HEIMDALL_TOOLS, MCP_SERVER_NAME, PERMISSION_MODE, TOOL_LOADER,
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
    assert "--output-format" in joined and "stream-json" in joined
    # `--no-session-persistence` used to be pinned here. It is deliberately gone:
    # the transcript it suppressed is the only per-model-call record this stack
    # can obtain, and what makes a turn independent is the absence of --resume,
    # not the absence of a file on disk. Verified on the live stack — see
    # test_a_stateless_turn_starts_cold_without_the_persistence_flag.
    assert "--resume" not in argv


def test_argv_pins_a_non_interactive_permission_mode(harness):
    """There is nobody on the other end of a Telegram bridge to approve a tool.

    Under the default mode a refusal reads "you haven't granted it yet", which
    invites the model to stop and ask the user — the exact dead end observed in
    trace 0ac81d7258d1cb0d. `dontAsk` states the refusal as final.
    """
    argv = harness.build_argv("q", config=AgentConfig(), system_suffix="s",
                              mcp_config_path="/tmp/mcp.json")
    assert argv[argv.index("--permission-mode") + 1] == PERMISSION_MODE
    assert PERMISSION_MODE == "dontAsk"


def test_argv_carries_the_configured_model(harness):
    config = AgentConfig(model_id="claude-haiku-4-5-20251001")
    argv = harness.build_argv("q", config=config, system_suffix="s",
                              mcp_config_path="/tmp/mcp.json")
    assert argv[argv.index("--model") + 1] == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------- harness note


def _named_in_note(note: str) -> set[str]:
    """Every `mcp__heimdall__<tool>` the note tells the agent to load."""
    return set(re.findall(rf"mcp__{MCP_SERVER_NAME}__(\w+)", note))


@pytest.mark.parametrize("config", [
    AgentConfig(),
    AgentConfig(tool_subset=("mcp_query",)),
    AgentConfig(tool_subset=("list_models", "get_overview")),
    AgentConfig(code_execution="allowed"),
])
def test_note_never_names_a_tool_the_matcher_will_refuse(harness, config):
    """The note is the agent's map of its own world; a wrong map is a trap.

    The note used to carry a hardcoded ToolSearch select-list including
    `get_overview`, which the default `tool_subset` does not grant. The agent
    loaded the schema, called the tool, was refused by the matcher, and — having
    no way to reach anyone — asked the user for permission instead of answering.
    """
    granted = {t.removeprefix(f"mcp__{MCP_SERVER_NAME}__")
               for t in harness.allowed_tools(config) if t.startswith("mcp__")}
    assert _named_in_note(harness.harness_note(config)) <= granted


def test_note_names_every_granted_tool(harness):
    """The other direction: a granted tool absent from the select-list is one
    the agent never learns it has, since MCP schemas are deferred."""
    config = AgentConfig()
    granted = {t.removeprefix(f"mcp__{MCP_SERVER_NAME}__")
               for t in harness.allowed_tools(config) if t.startswith("mcp__")}
    assert _named_in_note(harness.harness_note(config)) == granted


@pytest.mark.parametrize("config", [AgentConfig(),
                                    AgentConfig(code_execution="allowed")])
def test_note_forbids_asking_the_user_in_both_arms(harness, config):
    """Non-interactivity is a property of the channel, not of an arm.

    It lives here rather than in the registry prompt because the prompt is
    bootstrapped once — an existing deployment stays on `system_prompt@1` and
    would never receive the change — and because a harness fact belongs to the
    harness.
    """
    note = harness.harness_note(config)
    assert "Спрашивать некого" in note
    assert "не проси разрешений" in note.lower()
    # The turn is genuinely stateless — `run()` is handed the question and no
    # history, and `--no-session-persistence` forbids resuming. A clarifying
    # question is therefore not merely unhelpful, it is unanswerable.
    assert "Каждый ход самодостаточен" in note


# ------------------------------------------------------------ mcp config


def test_mcp_config_carries_the_acting_identity(harness):
    cfg = harness.mcp_config("2457060")
    env = cfg["mcpServers"][MCP_SERVER_NAME]["env"]
    assert env["HEIMDALL_EMPLOYEE_ID"] == "2457060"
    assert env["HEIMDALL_URL"] == "http://heimdall-emulator:8081"


def test_mcp_config_hides_tools_outside_the_subset(harness):
    """The advertised surface must equal the granted surface.

    MCP schemas are deferred, so an advertised-but-ungranted tool is one the
    agent discovers by name and burns a turn being refused on — and the refusal
    lands in `permission_denials`, which is supposed to mean "tried to leave the
    sandbox", not "tried a tool this condition happened to omit".
    """
    config = AgentConfig()
    env = harness.mcp_config("2457060", config)["mcpServers"][MCP_SERVER_NAME]["env"]
    served = set(env["HEIMDALL_TOOL_SUBSET"].split(","))
    assert served == set(config.tool_subset)
    assert "get_overview" not in served


def test_mcp_config_without_a_config_still_serves_everything(harness):
    """The bridge runs outside this stack too; absence must mean "no filter"."""
    env = harness.mcp_config("2457060")["mcpServers"][MCP_SERVER_NAME]["env"]
    assert "HEIMDALL_TOOL_SUBSET" not in env


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


# --------------------------------------------------------------- bridge
#
# The other half of the same contract, tested beside it on purpose: the harness
# and the bridge are what drifted apart, and a subset honoured in one and
# ignored in the other is exactly the failure this pair exists to catch.


BRIDGE = pathlib.Path(__file__).resolve().parents[1] / "heimdall" / "bridge.py"


def _bridge_tools(subset: str | None) -> list[str]:
    """Ask a real bridge process what it advertises."""
    env = {**os.environ, "HEIMDALL_URL": "http://127.0.0.1:1"}
    if subset is not None:
        env["HEIMDALL_TOOL_SUBSET"] = subset
    else:
        env.pop("HEIMDALL_TOOL_SUBSET", None)
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    proc = subprocess.run([sys.executable, str(BRIDGE)], input=request + "\n",
                          capture_output=True, text=True, env=env, timeout=30)
    if proc.returncode != 0:
        raise AssertionError(f"bridge exited {proc.returncode}: {proc.stderr[:400]}")
    body = json.loads(proc.stdout.splitlines()[0])
    return [t["name"] for t in body["result"]["tools"]]


def test_bridge_serves_everything_when_unconfigured():
    """Default must stay "all": the bridge also runs outside this stack."""
    assert set(_bridge_tools(None)) == set(HEIMDALL_TOOLS)


def test_bridge_serves_only_the_subset():
    names = _bridge_tools("list_models,mcp_query")
    assert set(names) == {"list_models", "mcp_query"}


def test_bridge_refuses_an_unknown_tool_name_in_the_subset():
    """A typo'd name that silently narrowed the surface would be a run claiming
    a condition it did not apply — the same argument AgentConfig already makes."""
    with pytest.raises(AssertionError, match="неизвестные инструменты"):
        _bridge_tools("mcp_query,mcp_qeury")


def test_harness_subset_is_a_vocabulary_the_bridge_accepts(harness):
    """Whatever the harness puts in the environment, the bridge must serve."""
    config = AgentConfig()
    env = harness.mcp_config("1", config)["mcpServers"][MCP_SERVER_NAME]["env"]
    assert set(_bridge_tools(env["HEIMDALL_TOOL_SUBSET"])) == set(config.tool_subset)


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


# ------------------------------------------------- conversation_mode


CONVERSING = AgentConfig(conversation_mode="resume")


def test_stateless_is_the_default():
    """A batch arm must not silently start sharing context between turns.

    Runs that carry each other's history are not independent samples, and the
    per-turn token counts stop meaning what the RQ2 tables say they mean.
    """
    assert AgentConfig().conversation_mode == "stateless"


def test_resume_turns_off_session_persistence(harness):
    """`--no-session-persistence` and `--resume` are mutually exclusive: the
    flag forbids writing the transcript the other one needs to read."""
    argv = harness.build_argv("q", config=CONVERSING, system_suffix="s",
                              mcp_config_path="/tmp/mcp.json")
    assert "--no-session-persistence" not in argv


def test_first_turn_of_a_conversation_does_not_resume(harness):
    """There is nothing to resume yet, and `--resume` with no session would
    fail the turn rather than start one."""
    argv = harness.build_argv("q", config=CONVERSING, system_suffix="s",
                              mcp_config_path="/tmp/mcp.json",
                              resume_session_id=None)
    assert "--resume" not in argv


def test_later_turns_resume_the_bound_session(harness):
    argv = harness.build_argv("q", config=CONVERSING, system_suffix="s",
                              mcp_config_path="/tmp/mcp.json",
                              resume_session_id="abc-123")
    assert argv[argv.index("--resume") + 1] == "abc-123"


def test_a_stateless_config_never_resumes(harness):
    """Defence in depth against a caller passing a stale id: the mode decides,
    not the presence of an id.

    This is now the *whole* mechanism keeping stateless turns independent, so it
    matters more than it did when `--no-session-persistence` stood behind it.
    """
    argv = harness.build_argv("q", config=AgentConfig(), system_suffix="s",
                              mcp_config_path="/tmp/mcp.json",
                              resume_session_id="abc-123")
    assert "--resume" not in argv


def test_resumable_sessions_get_a_stable_workdir(harness):
    """Claude Code files a transcript under the project it was started in, so
    two turns of one conversation must run in the same directory."""
    first = harness.session_workdir(CONVERSING, "ses_abc")
    second = harness.session_workdir(CONVERSING, "ses_abc")
    assert first == second
    assert harness.session_workdir(CONVERSING, "ses_other") != first


def test_stateless_sessions_get_a_throwaway_workdir(harness):
    """None means mkdtemp: no transcript is left behind to resume."""
    assert harness.session_workdir(AgentConfig(), "ses_abc") is None


def test_session_workdir_does_not_escape_its_root(harness):
    """The id comes from the store, but it is used to build a path."""
    evil = harness.session_workdir(CONVERSING, "../../etc/passwd")
    assert evil is not None
    assert ".." not in evil.parts
    assert str(evil).startswith(harness.session_root)


# ------------------------------------------------ the dialogue note


def test_dialogue_note_replaces_the_stateless_one(harness):
    """A resumable session told "истории у тебя нет" is being lied to, and the
    lie costs exactly the follow-up questions this mode exists to allow."""
    note = harness.harness_note(CONVERSING)
    assert "Каждый ход самодостаточен" not in note
    assert "Это диалог" in note


@pytest.mark.parametrize("config", [
    AgentConfig(),
    CONVERSING,
    AgentConfig(code_execution="allowed"),
    AgentConfig(code_execution="allowed", conversation_mode="resume"),
])
def test_permission_refusal_is_final_in_every_mode(harness, config):
    """The bd0d86d fix is about who can grant a scope, not about how many turns
    the session runs for. A researcher reading answers in Telegram cannot widen
    a Heimdall permission however long the conversation lasts.
    """
    note = harness.harness_note(config)
    assert "не проси разрешений" in note.lower()
    assert "Отказ инструмента окончателен" in note


@pytest.mark.parametrize("config", [
    AgentConfig(),
    CONVERSING,
    AgentConfig(code_execution="allowed"),
    AgentConfig(code_execution="allowed", conversation_mode="resume"),
])
def test_note_never_names_a_refused_tool_in_any_mode(harness, config):
    granted = {t.removeprefix(f"mcp__{MCP_SERVER_NAME}__")
               for t in harness.allowed_tools(config) if t.startswith("mcp__")}
    assert _named_in_note(harness.harness_note(config)) <= granted


# ---------------------------------------------------- AskUserQuestion


def test_ask_user_question_stays_denied_in_every_mode(harness):
    """Probed on Claude Code 2.1.220: headless `-p` has no AskUserQuestion at
    all. The `init` event enumerates the whole surface, deferred tools included,
    and the name is absent even when passed to --allowed-tools — under both
    --input-format text and --input-format stream-json.

    So granting it would add a name the session cannot resolve. Under `resume`
    the turn boundary is the asking mechanism instead: the agent ends a turn
    with a question and the researcher answers in the next message.

    If a later CLI does expose it, this test is the place that should fail.
    """
    assert "AskUserQuestion" in DENIED_TOOLS
    for config in (AgentConfig(), CONVERSING,
                   AgentConfig(code_execution="allowed"),
                   AgentConfig(code_execution="allowed",
                               conversation_mode="resume")):
        assert "AskUserQuestion" in harness.denied_tools(config)
        assert "AskUserQuestion" not in harness.allowed_tools(config)
