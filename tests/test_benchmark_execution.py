"""Unit tests for mode activation and trace-derived observations."""
from __future__ import annotations

import json

import pytest

from sim.benchmark.execution import (
    AgentRequest, AgentTurn, PinnedConfigActivator, StandSessionExecutor,
    fatal_turn_error, trace_observations,
)
from sim.benchmark.modes import (
    GENERAL_KNOWLEDGE_TOOLS, SKILL_TOOLS, BenchmarkMode, CommonConditions,
    ModeConfig,
)


def mode(name: str = "existing_skills", *, mock: bool = False) -> ModeConfig:
    common = CommonConditions(
        "model", 0.0, "prompt@1", "heimdall-sandbox@test", True, "instant", ()
    )
    enabled = name != BenchmarkMode.GENERAL_KNOWLEDGE
    generated_names = (
        ("generated_headcount",)
        if not mock and name == BenchmarkMode.GENERATED_SKILLS else ()
    )
    return ModeConfig(
        name, SKILL_TOOLS if enabled else GENERAL_KNOWLEDGE_TOOLS,
        None, None, None, None,
        generated_names, common, skills_enabled=enabled, is_mock=mock,
    )


def _agent_config(tools: tuple[str, ...]) -> dict:
    return {
        "model_id": "model",
        "temperature": 0.0,
        "tool_subset": list(tools),
        "code_execution": "forbidden",
        "conversation_mode": "stateless",
    }


def test_pinned_activator_rejects_floating_refs_and_real_generated_mode() -> None:
    with pytest.raises(ValueError, match="pinned config_ref"):
        PinnedConfigActivator({"existing_skills": "agent"}).activate(mode())
    with pytest.raises(ValueError, match="config_reader"):
        PinnedConfigActivator({"existing_skills": "agent@1"}).activate(mode())
    with pytest.raises(ValueError, match="tool_subset differs"):
        PinnedConfigActivator(
            {"general_knowledge": "agent@1"},
            config_reader=lambda _ref: _agent_config(SKILL_TOOLS),
        ).activate(mode("general_knowledge"))
    with pytest.raises(ValueError, match="catalog-mounting"):
        PinnedConfigActivator(
            {"generated_skills": "agent@1"},
            config_reader=lambda _ref: _agent_config(SKILL_TOOLS),
        ).activate(mode("generated_skills"))


def test_trace_observations_extract_calls_skills_errors_and_time() -> None:
    trace = {"tree": [{
        "name": "agent.turn",
        "attributes": {
            "b2e.turn.duration_ms": 40,
            "b2e.turn.tool_time_ms": 12,
            "b2e.turn.api_duration_ms": 30,
            "b2e.turn.ttft_ms": 5,
            "b2e.turn.ttft_stream_ms": 4,
            "b2e.turn.time_to_request_ms": 2,
            "b2e.turn.iterations": 2,
            "b2e.turn.uncached_prompt_tokens": 10,
            "b2e.turn.cost_usd": 0.02,
            "b2e.permission_denials": 1,
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 20,
            "llm.token_count.total": 120,
            "llm.token_count.prompt_details.cache_read": 80,
            "llm.token_count.prompt_details.cache_write": 10,
        },
        "children": [
            {
                "name": "mcp__heimdall__find_skills",
                "attributes": {
                    "tool.name": "mcp__heimdall__find_skills",
                    "output.value": json.dumps({
                        "results": [{"name": "generated_headcount"}]
                    }),
                },
            },
            {
                "name": "mcp__heimdall__get_skill",
                "attributes": {
                    "tool.name": "mcp__heimdall__get_skill",
                    "input.value": json.dumps({"name": "generated_headcount"}),
                },
            },
            {
                "name": "heimdall.mcp_query",
                "status": {"status_code": "ERROR"},
                "attributes": {
                    "b2e.heimdall.endpoint": "mcp_query",
                    "b2e.http.status": 400,
                    "b2e.heimdall.error_code": "bad_request",
                    "b2e.heimdall.rows": 0,
                    "b2e.heimdall.response_bytes": 256,
                },
            },
        ],
    }]}
    observed = trace_observations(AgentTurn(
        "", {"tool_calls": 3, "heimdall_calls": 3, "iterations": 2,
             "prompt_tokens": 100, "completion_tokens": 20,
             "total_tokens": 120, "cost_usd": 0.02, "latency_ms": 55}, trace,
    ))
    assert observed["found_skills"] == ["generated_headcount"]
    assert observed["loaded_skills"] == ["generated_headcount"]
    assert observed["mcp_query_calls"] == 1
    assert observed["failed_tool_calls"] == 1
    assert observed["http_statuses"] == [400]
    assert observed["error_codes"] == ["bad_request"]
    assert observed["mcp_query_rows"] == [0]
    assert observed["agent_duration_ms"] == 40
    assert observed["tool_time_ms"] == 12
    assert observed["api_duration_ms"] == 30
    assert observed["ttft_ms"] == 5
    assert observed["prompt_tokens"] == 100
    assert observed["uncached_prompt_tokens"] == 10
    assert observed["cache_read_tokens"] == 80
    assert observed["cache_creation_tokens"] == 10
    assert observed["completion_tokens"] == 20
    assert observed["cache_hit_ratio"] == 0.8
    assert observed["cost_usd"] == 0.02
    assert observed["permission_denials"] == 1
    assert observed["http_error_count"] == 1
    assert observed["heimdall_response_bytes"] == 256
    assert observed["tool_time_ratio"] == 0.3


def test_fatal_turn_error_ignores_successful_harness_denials() -> None:
    answer = '{"result": [{"grade": 7, "employee_count": 1}], "message": null}'
    assert fatal_turn_error("['denied:Bash']", answer=answer) is None
    assert fatal_turn_error("['denied:Bash', 'denied:Write']", answer=answer) is None
    assert fatal_turn_error(
        "['denied:mcp__heimdall__find_skills', "
        "'denied:mcp__heimdall__mcp_query']",
        answer=answer,
    ) is None
    assert fatal_turn_error("['denied:Bash']", answer="  ") == "['denied:Bash']"
    assert fatal_turn_error("['denied:Bash', 'HTTP 502']", answer=answer) == (
        "['denied:Bash', 'HTTP 502']"
    )
    assert fatal_turn_error("trace unavailable for session ses-1", answer=answer) == (
        "trace unavailable for session ses-1"
    )


def test_stand_executor_leaves_missing_trace_to_the_runner(monkeypatch) -> None:
    from sim.skill_eval.types import TurnResult

    class TraceMissingStand:
        def __init__(self, **_options) -> None:
            pass

        def run(self, _case, _spec) -> TurnResult:
            return TurnResult(
                session_id="ses-target", answer="{}", stats={}, trace=None,
                error=None, retrieved_skills=[], heimdall_calls=4,
                tool_calls=4, total_tokens=0, latency_ms=1.0,
                trace_id=None, root_span_id=None,
            )

    monkeypatch.setattr("sim.skill_eval.stand.StandClient", TraceMissingStand)
    executor = StandSessionExecutor(trace_attempts=1)

    turn = executor.execute(AgentRequest(
        query="question", employee_id="1", config_ref="agent@1",
        metadata={"case_id": "case-1", "heimdall_access": "disabled"},
    ))

    assert turn.error is None
    assert turn.trace is None
    assert turn.session_id == "ses-target"
    assert turn.stats["heimdall_calls"] == 0
