"""OpenCode harness configuration and JSON-stream accounting."""
from __future__ import annotations

import json
import re

import pytest

from sim import telemetry
from sim.agent.claude_code import ClaudeCodeHarness, emit_spans
from sim.agent.config import AgentConfig
from sim.agent.opencode import (
    OpenCodeHarness,
    api_token_prices_from_env,
    parse_stream,
)
from sim.agent.provider import runtime_agent_config


def harness() -> OpenCodeHarness:
    return OpenCodeHarness(heimdall_url="http://heimdall:8081", heimdall_token="t")


@pytest.mark.parametrize("tool_subset,skill_enabled", [
    pytest.param((), False, id="general_knowledge"),
    pytest.param(("list_models", "describe_model", "mcp_query"), False,
                 id="skills_disabled"),
    pytest.param(("get_overview",), True, id="overview"),
    pytest.param(("get_docs",), True, id="docs"),
    pytest.param(("find_skills",), True, id="find_skills"),
    pytest.param(("get_skill",), True, id="get_skill"),
])
@pytest.mark.parametrize("code_execution", ["forbidden", "allowed"])
@pytest.mark.parametrize("conversation_mode", ["stateless", "resume"])
def test_harness_notes_share_policy_but_not_tool_discovery(
    tool_subset, skill_enabled, code_execution, conversation_mode, monkeypatch,
):
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    config = AgentConfig(harness="open_code", tool_subset=tool_subset,
                         code_execution=code_execution, conversation_mode=conversation_mode)
    kwargs = dict(heimdall_url="http://unused", heimdall_token="test",
                  runner_path="/custom/approved-runner")
    opencode = OpenCodeHarness(**kwargs)
    claude = ClaudeCodeHarness(**kwargs)
    note = opencode.harness_note(config)
    claude_note = claude.harness_note(config)

    # Only the discovery paragraph may differ; all capability and channel rules match.
    discovery = r"\* Инструменты Heimdall[^\n]*(?:\n  [^\n]*)*\n"
    assert re.sub(discovery, "", note) == re.sub(discovery, "", claude_note)
    assert "ToolSearch" not in note
    assert "mcp__heimdall__" not in note
    assert set(re.findall(r"\bheimdall_\w+", note)) == {
        name for name, grant in opencode._permissions(config).items()
        if name.startswith("heimdall_") and grant == "allow"
    }
    assert ("ToolSearch" in claude_note) == bool(tool_subset)
    assert "Данные — только через инструменты Heimdall" in note
    assert "Отказ в доступе (403)" in note
    assert "Не проси разрешений" in note
    assert ("Это диалог" in note) == (conversation_mode == "resume")
    assert ("Каждый ход самодостаточен" in note) == (conversation_mode == "stateless")
    if code_execution == "allowed":
        assert "разрешено писать код" in note
        assert "не можешь" not in note
    else:
        assert "**не можешь** написать и выполнить код" in note
        assert ("Канал навыков отключён" in note) == (not skill_enabled)
        assert (opencode.runner_path in note) == skill_enabled
        if skill_enabled:
            assert f"{opencode.runner_path} <sha256> '<json>'" in note
    built = opencode.build_config(config=config, system_prompt="system",
                                 employee_id="42", trace_log="/tmp/unused")
    assert built["agent"]["b2e"]["prompt"] == "system\n" + note


@pytest.mark.parametrize("tool_subset,skill_enabled", [
    pytest.param((), False, id="general_knowledge"),
    pytest.param(("list_models", "describe_model", "mcp_query"), False,
                 id="skills_disabled"),
    pytest.param(("get_overview",), True, id="overview"),
    pytest.param(("get_docs",), True, id="docs"),
    pytest.param(("find_skills",), True, id="find_skills"),
    pytest.param(("get_skill",), True, id="get_skill"),
])
@pytest.mark.parametrize("code_execution", ["forbidden", "allowed"])
def test_bash_permission_respects_skill_channel(
    tool_subset, skill_enabled, code_execution,
):
    runner = harness()
    permissions = runner._permissions(AgentConfig(
        harness="open_code", tool_subset=tool_subset,
        code_execution=code_execution,
    ))

    assert permissions["*"] == "deny"
    if code_execution == "allowed":
        assert permissions["bash"] == "allow"
    elif skill_enabled:
        assert permissions["bash"] == {"*": "deny", f"{runner.runner_path} *": "allow"}
    else:
        assert "bash" not in permissions


def test_environment_is_the_runtime_source_of_model_and_harness(monkeypatch):
    monkeypatch.setenv("B2E_MODEL", "glm-env")
    monkeypatch.setenv("B2E_HARNESS", "open_code")
    config = runtime_agent_config(
        AgentConfig(model_id="registry-model", harness="claude_code").as_dict()
    )
    assert config.model_id == "glm-env"
    assert config.harness == "open_code"


def test_config_exposes_only_the_selected_heimdall_tools(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    config = AgentConfig(
        harness="open_code", tool_subset=("list_models", "mcp_query")
    )
    built = harness().build_config(
        config=config, system_prompt="system", employee_id="42",
        trace_log="/tmp/trace.jsonl",
    )
    permission = built["agent"]["b2e"]["permission"]
    assert permission["*"] == "deny"
    assert permission["heimdall_list_models"] == "allow"
    assert permission["heimdall_mcp_query"] == "allow"
    assert "heimdall_get_docs" not in permission
    assert built["mcp"]["heimdall"]["command"][0] == "python3.12"
    assert built["model"].startswith("zai-coding-plan/")
    assert "steps" not in built["agent"]["b2e"]
    assert "Каждый ход самодостаточен" in built["agent"]["b2e"]["prompt"]


def test_resumable_config_describes_a_real_dialogue(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    built = harness().build_config(
        config=AgentConfig(harness="open_code", conversation_mode="resume"),
        system_prompt="system", employee_id="42", trace_log="/tmp/trace.jsonl",
    )
    assert "Это диалог" in built["agent"]["b2e"]["prompt"]
    assert "Каждый ход самодостаточен" not in built["agent"]["b2e"]["prompt"]


def test_json_stream_preserves_answer_tools_and_usage():
    rows = [
        {"type": "step_start", "timestamp": 1000, "sessionID": "ses_1",
         "part": {"type": "step-start", "messageID": "msg_1"}},
        {"type": "reasoning", "timestamp": 1050, "sessionID": "ses_1",
         "part": {"type": "reasoning", "messageID": "msg_1",
                  "text": "need a tool"}},
        {"type": "text", "timestamp": 1060, "sessionID": "ses_1",
         "part": {"type": "text", "messageID": "msg_1",
                  "text": "checking"}},
        {"type": "tool_use", "timestamp": 1100, "sessionID": "ses_1",
         "part": {"type": "tool", "messageID": "msg_1", "callID": "call_1",
                  "tool": "heimdall_mcp_query",
                  "state": {"status": "completed", "input": {"limit": 1},
                            "output": "one row"}}},
        {"type": "step_finish", "timestamp": 1200, "sessionID": "ses_1",
         "part": {"type": "step-finish", "messageID": "msg_1",
                  "reason": "tool-calls", "cost": 0.01,
                  "tokens": {"input": 100, "output": 20, "reasoning": 7,
                             "cache": {"read": 300, "write": 40}}}},
        {"type": "step_start", "timestamp": 1300, "sessionID": "ses_1",
         "part": {"type": "step-start", "messageID": "msg_2"}},
        {"type": "reasoning", "timestamp": 1350, "sessionID": "ses_1",
         "part": {"type": "reasoning", "messageID": "msg_2",
                  "text": "now answer"}},
        {"type": "text", "timestamp": 1400, "sessionID": "ses_1",
         "part": {"type": "text", "messageID": "msg_2", "text": "answer"}},
        {"type": "step_finish", "timestamp": 1500, "sessionID": "ses_1",
         "part": {"type": "step-finish", "messageID": "msg_2",
                  "reason": "stop", "cost": 0.02,
                  "tokens": {"input": 110, "output": 30, "reasoning": 9,
                             "cache": {"read": 400, "write": 50}}}},
    ]
    result = parse_stream(
        "\n".join(json.dumps(row) for row in rows),
        model="zai-coding-plan/glm", duration_ms=600,
    )
    assert result.answer == "answer"
    assert result.session_id == "ses_1"
    assert result.num_turns == 2
    assert result.prompt_tokens == 1000
    assert result.output_tokens == 50
    assert result.reasoning_tokens == 16
    assert result.completion_tokens == 66
    assert result.total_tokens == 1066
    assert result.cost_usd == 0.03
    assert result.cost_mode == "reported"
    assert result.heimdall_calls == 1
    assert result.tool_calls[0]["name"] == "mcp__heimdall__mcp_query"
    assert len(result.llm_calls) == 2
    assert result.llm_calls[0].contents == [
        {"type": "reasoning", "text": "need a tool"},
        {"type": "text", "text": "checking"},
    ]


def test_subscription_cost_is_unknown_without_explicit_prices():
    row = {
        "type": "step_finish", "timestamp": 1000,
        "part": {"messageID": "msg_1", "cost": 0,
                 "tokens": {"input": 100, "output": 20, "reasoning": 5,
                            "cache": {"read": 300, "write": 40}}},
    }
    result = parse_stream(json.dumps(row), model="zai-coding-plan/glm")
    assert result.cost_usd is None
    assert result.cost_mode == "subscription"


def test_explicit_prices_produce_api_equivalent_estimate():
    row = {
        "type": "step_finish", "timestamp": 1000,
        "part": {"messageID": "msg_1", "cost": 0,
                 "tokens": {"input": 100, "output": 20, "reasoning": 5,
                            "cache": {"read": 300, "write": 40}}},
    }
    result = parse_stream(
        json.dumps(row), model="zai-coding-plan/glm",
        token_prices={"input": 1.0, "output": 2.0,
                      "cache_read": 0.1, "cache_write": 1.25},
    )
    assert result.cost_usd == pytest.approx(0.00023)
    assert result.cost_mode == "api_estimate"


def test_partial_api_price_configuration_is_rejected(monkeypatch):
    for variable in (
        "B2E_API_PRICE_INPUT_USD_PER_MTOK",
        "B2E_API_PRICE_OUTPUT_USD_PER_MTOK",
        "B2E_API_PRICE_CACHE_READ_USD_PER_MTOK",
        "B2E_API_PRICE_CACHE_WRITE_USD_PER_MTOK",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("B2E_API_PRICE_INPUT_USD_PER_MTOK", "1")
    with pytest.raises(ValueError, match="requires all four rates"):
        api_token_prices_from_env()


@pytest.mark.parametrize("capture_content", [True, False])
def test_reasoning_and_unknown_subscription_cost_reach_spans(
    spans, fingerprint, monkeypatch, capture_content,
):
    monkeypatch.setenv("B2E_TRACE_LLM_CONTENT", "1" if capture_content else "0")
    rows = [
        {"type": "step_start", "timestamp": 1000,
         "part": {"messageID": "msg_1"}},
        {"type": "reasoning", "timestamp": 1010,
         "part": {"messageID": "msg_1", "text": "think"}},
        {"type": "text", "timestamp": 1020,
         "part": {"messageID": "msg_1", "text": "answer"}},
        {"type": "step_finish", "timestamp": 1030,
         "part": {"messageID": "msg_1", "cost": 0,
                  "tokens": {"input": 10, "output": 3, "reasoning": 2,
                             "cache": {"read": 20, "write": 0}}}},
    ]
    result = parse_stream("\n".join(map(json.dumps, rows)),
                          model="zai-coding-plan/glm")
    assert result.llm_calls[0].input_messages is None
    result.system_suffix = "known harness instructions"
    with telemetry.start_run(
        "b2e.turn", fingerprint=fingerprint,
        session_id="ses_test", employee_id="employee_test",
    ) as root:
        emit_spans(result, root=root, harness_name="open_code")

    root = next(span for span in spans.get_finished_spans()
                if span.name == "b2e.turn")
    llm = next(span for span in spans.get_finished_spans()
               if span.name == "llm.messages.create")
    assert root.attributes["b2e.cost.mode"] == "subscription"
    assert "b2e.turn.cost_usd" not in root.attributes
    assert root.attributes["b2e.turn.reasoning_tokens"] == 2
    assert not any(key.startswith("llm.input_messages") for key in llm.attributes)
    if capture_content:
        assert llm.attributes["b2e.llm.prompt_reconstruction"] == "unavailable"
        assert llm.attributes["b2e.llm.prompt_missing"] == (
            "conversation,cli_system_prompt,tool_schemas")
        assert llm.attributes["llm.system"] == "known harness instructions"
        assert llm.attributes["b2e.llm.system_partial"] is True
        assert llm.attributes[
            "llm.output_messages.0.message.contents.0.message_content.type"
        ] == "reasoning"
    else:
        assert llm.attributes["b2e.trace.llm_content"] == "disabled"
        assert "b2e.llm.prompt_reconstruction" not in llm.attributes
        assert "llm.system" not in llm.attributes
        assert not any(key.startswith("llm.output_messages") for key in llm.attributes)
    assert llm.attributes[
        "llm.token_count.completion_details.reasoning"
    ] == 2
    assert llm.attributes["b2e.llm.protocol"] == "openai"
    assert "llm.cost.total" not in llm.attributes
