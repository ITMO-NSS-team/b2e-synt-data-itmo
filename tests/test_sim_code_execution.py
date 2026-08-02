"""The code-execution arm: a switchable rival to the project's own premise.

`code_execution="allowed"` lets the agent write and run Python in-session, so a
researcher can test the obvious objection — that the no-code constraint is what
costs the agent its accuracy on the question basket — instead of assuming either
answer.

The tests that matter most here are the ones asserting the DEFAULT is unchanged.
A degree of freedom that silently drifts the control condition would invalidate
every measurement taken before it was added.
"""
from __future__ import annotations

import pytest

from sim.agent.claude_code import (
    ALWAYS_DENIED, CODE_EXECUTION_TOOLS, DENIED_TOOLS, ClaudeCodeHarness,
)
from sim.agent.config import CODE_EXECUTION_MODES, AgentConfig


@pytest.fixture()
def harness():
    return ClaudeCodeHarness(
        heimdall_url="http://heimdall-emulator:8081",
        heimdall_token="tok",
        bridge_path="/app/heimdall/bridge.py",
        runner_path="/opt/skills/run",
    )


FORBIDDEN = AgentConfig()
ALLOWED = AgentConfig(code_execution="allowed")


# ------------------------------------------------------- the default holds


def test_default_is_the_status_quo():
    assert AgentConfig().code_execution == "forbidden"


def test_default_tool_policy_is_byte_for_byte_unchanged(harness):
    """The control arm must not move because a new arm was added."""
    allowed = harness.allowed_tools(FORBIDDEN)
    assert "Bash" not in allowed
    assert "Bash(/opt/skills/run:*)" in allowed
    assert harness.denied_tools(FORBIDDEN) == list(DENIED_TOOLS)
    for tool in ("Read", "Glob", "Grep", "Write", "Edit"):
        assert tool in harness.denied_tools(FORBIDDEN)


def test_default_prompt_still_says_code_is_impossible(harness):
    note = harness.harness_note(FORBIDDEN)
    assert "не можешь" in note and "/opt/skills/run" in note


# ------------------------------------------------------------ the new arm


def test_allowed_grants_unqualified_bash(harness):
    """A narrowed Bash would be a straw man that loses for the wrong reason."""
    allowed = harness.allowed_tools(ALLOWED)
    assert "Bash" in allowed
    assert "Bash(/opt/skills/run:*)" not in allowed


def test_allowed_grants_the_tools_the_arm_actually_needs(harness):
    allowed = harness.allowed_tools(ALLOWED)
    for tool in CODE_EXECUTION_TOOLS:
        assert tool in allowed, tool


def test_allowed_lifts_those_tools_from_the_deny_list(harness):
    denied = harness.denied_tools(ALLOWED)
    assert not (set(CODE_EXECUTION_TOOLS) & set(denied))


def test_allowed_still_denies_egress_and_subagents(harness):
    """Network and sub-agents change WHAT the agent can reach, not WHETHER it
    can compute — leaving them on would confound the comparison."""
    denied = set(harness.denied_tools(ALLOWED))
    for tool in ALWAYS_DENIED:
        if tool in DENIED_TOOLS:
            assert tool in denied, tool


def test_allowed_prompt_tells_the_agent_it_may_compute(harness):
    """A prompt that forbids what the policy permits turns the arm into a test
    of prompt compliance rather than of capability."""
    note = harness.harness_note(ALLOWED)
    assert "разрешено писать код" in note
    assert "не можешь" not in note


def test_argv_carries_the_arm(harness):
    for config, expect_bare_bash in ((FORBIDDEN, False), (ALLOWED, True)):
        argv = harness.build_argv("q", config=config, system_suffix="s",
                                  mcp_config_path="/tmp/mcp.json")
        allowed = argv[argv.index("--allowed-tools") + 1].split(",")
        assert ("Bash" in allowed) is expect_bare_bash


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("mode", CODE_EXECUTION_MODES)
def test_valid_modes_are_accepted(mode):
    harness_name = "claude_code"
    assert AgentConfig(code_execution=mode,
                       harness=harness_name).code_execution == mode


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="code_execution"):
        AgentConfig(code_execution="sometimes")


def test_allowed_requires_the_claude_code_harness():
    """messages_api exposes no tool that can run code. Silently accepting this
    would produce an arm labelled 'code allowed' that is really the control."""
    with pytest.raises(ValueError, match="requires harness='claude_code'"):
        AgentConfig(code_execution="allowed", harness="messages_api")


def test_forbidden_works_with_either_harness():
    for harness_name in ("claude_code", "messages_api"):
        assert AgentConfig(harness=harness_name).code_execution == "forbidden"


# ------------------------------------------------------- comparability


def test_the_two_arms_are_different_conditions(tmp_path):
    """Two runs differing only in this flag must not share a condition_id, or
    the comparison would treat the arms as one condition with noise."""
    from sim.fingerprint import RunFingerprint
    from sim.registry import Registry

    registry = Registry(tmp_path / "r.db")
    try:
        control = registry.commit("agent_config", "agent",
                                  AgentConfig().as_dict(), actor="t")
        treatment = registry.commit("agent_config", "agent",
                                    ALLOWED.as_dict(), actor="t")
        assert control.version != treatment.version

        base = dict(
            prompt_registry_version="system_prompt@1",
            skill_registry_hash="sha256:" + "00" * 32,
            model_id="claude-haiku-4-5-20251001", temperature=0.0,
            data_snapshot_hash="snap@1", traps_enabled=True,
            latency_profile="realistic")
        a = RunFingerprint.create(agent_config_version=control.ref, **base)
        b = RunFingerprint.create(agent_config_version=treatment.ref, **base)
        assert a.condition_id != b.condition_id
    finally:
        registry.close()


def test_the_flag_round_trips_through_the_config_store():
    restored = AgentConfig.from_dict(ALLOWED.as_dict())
    assert restored.code_execution == "allowed"
    assert restored.harness == "claude_code"
