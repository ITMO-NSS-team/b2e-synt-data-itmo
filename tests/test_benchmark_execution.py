"""Unit tests for mode activation and trace-derived observations."""
from __future__ import annotations

import json

import pytest

from sim.benchmark.execution import (
    AgentRequest, AgentTurn, PinnedConfigActivator, StandSessionExecutor,
    fatal_turn_error, trace_observations,
)
from sim.benchmark.modes import DATA_TOOLS, SKILL_TOOLS, BenchmarkMode, CommonConditions, ModeConfig


def mode(name: str = "existing_skills", *, mock: bool = False) -> ModeConfig:
    common = CommonConditions(
        "model", 0.0, "prompt@1", "heimdall-sandbox@test", True, "instant", ()
    )
    enabled = name != BenchmarkMode.SKILLS_DISABLED
    generated_names = (
        ("generated_headcount",)
        if not mock and name == BenchmarkMode.GENERATED_SKILLS else ()
    )
    return ModeConfig(
        name, SKILL_TOOLS if enabled else DATA_TOOLS, None, None, None, None,
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
            {"skills_disabled": "agent@1"},
            config_reader=lambda _ref: _agent_config(SKILL_TOOLS),
        ).activate(mode("skills_disabled"))
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
                },
            },
        ],
    }]}
    observed = trace_observations(AgentTurn(
        "", {"tool_calls": 3, "heimdall_calls": 3, "total_tokens": 99,
             "latency_ms": 55}, trace,
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


def test_fatal_turn_error_ignores_successful_harness_denials() -> None:
    answer = '{"result": [{"grade": 7, "employee_count": 1}], "message": null}'
    assert fatal_turn_error("['denied:Bash']", answer=answer) is None
    assert fatal_turn_error("['denied:Bash', 'denied:Write']", answer=answer) is None
    assert fatal_turn_error("['denied:Bash']", answer="  ") == "['denied:Bash']"
    assert fatal_turn_error("['denied:Bash', 'HTTP 502']", answer=answer) == (
        "['denied:Bash', 'HTTP 502']"
    )
    assert fatal_turn_error("trace unavailable for session ses-1", answer=answer) == (
        "trace unavailable for session ses-1"
    )


def test_stand_executor_marks_missing_session_trace_as_error(monkeypatch) -> None:
    from sim.skill_eval.types import TurnResult

    class TraceMissingStand:
        def __init__(self, **_options) -> None:
            pass

        def run(self, _case, _spec) -> TurnResult:
            return TurnResult(
                session_id="ses-target", answer="{}", stats={}, trace=None,
                error=None, retrieved_skills=[], heimdall_calls=0,
                tool_calls=0, total_tokens=0, latency_ms=1.0,
                trace_id=None, root_span_id=None,
            )

    monkeypatch.setattr("sim.skill_eval.stand.StandClient", TraceMissingStand)
    executor = StandSessionExecutor(trace_attempts=1)

    turn = executor.execute(AgentRequest(
        query="question", employee_id="1", config_ref="agent@1",
        metadata={"case_id": "case-1"},
    ))

    assert turn.error == "trace unavailable for session ses-target"
