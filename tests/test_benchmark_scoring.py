"""Unit tests for deterministic answer normalization and metrics."""
from __future__ import annotations

import json
from pathlib import Path

from sim.benchmark.cases import BenchmarkCase
from sim.benchmark.modes import (
    GENERAL_KNOWLEDGE_TOOLS, SKILL_TOOLS, BenchmarkMode, CommonConditions,
    ModeConfig,
)
from sim.benchmark.scoring import NormalizedAnswer, calculate_metrics, normalize_answer
from tests.fixtures.constants import EXAMPLE


def ready_case() -> BenchmarkCase:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update({
        "case_id": "case-ready", "status": "ready", "employee_id": "123",
        "employee_role": "manager", "snapshot_id": "heimdall-sandbox@test",
        "evaluation_contract": {
            "expected_outcome": "answer",
            "gold_result": [
                {"grade": 10, "employee_count": 3},
                {"grade": 11, "employee_count": 2},
            ],
            "comparison": {
                "ordered": False, "row_key": ["grade"],
                "allow_extra_rows": False, "numeric_absolute_tolerance": 0,
            },
        },
    })
    return BenchmarkCase(Path("/authorial/case-ready.json"), raw)


def mode(name: str = "existing_skills") -> ModeConfig:
    common = CommonConditions(
        "model", 0.0, "prompt@1", "heimdall-sandbox@test", True, "instant", ()
    )
    enabled = name != BenchmarkMode.GENERAL_KNOWLEDGE
    generated_names = ("generated_headcount",) if name == BenchmarkMode.GENERATED_SKILLS else ()
    return ModeConfig(
        name, SKILL_TOOLS if enabled else GENERAL_KNOWLEDGE_TOOLS,
        None, None, None, None,
        generated_names, common, skills_enabled=enabled,
    )


def test_normalizer_extracts_fenced_json_and_rejects_prose() -> None:
    case = ready_case()
    fence = "\x60\x60\x60"
    expected = {
        "result": case.raw["evaluation_contract"]["gold_result"],
        "message": None,
    }
    answer = (
        f"Результат:\n{fence}json\n"
        + json.dumps(expected, ensure_ascii=False)
        + f"\n{fence}"
    )
    normalized = normalize_answer(answer, case.raw["response_contract"])
    assert normalized.value == expected
    assert normalized.error is None
    missing = normalize_answer("Получилось пять сотрудников.", case.raw["response_contract"])
    assert missing.value is None
    assert "no JSON object" in missing.error


def test_metrics_respect_order_tolerance_and_generated_routing() -> None:
    case = ready_case()
    case.raw["expected_skills"] = ["generated_headcount"]
    case.raw["evaluation_contract"]["comparison"]["numeric_absolute_tolerance"] = 1
    actual = NormalizedAnswer({
        "result": [
            {"grade": 11, "employee_count": 3},
            {"grade": 10, "employee_count": 2},
        ],
        "message": None,
    }, None)
    observations = {
        "loaded_skills": ["generated_headcount"], "heimdall_calls": 3,
        "mcp_query_calls": 1, "failed_tool_calls": 0, "total_tokens": 10,
        "latency_ms": 20, "agent_duration_ms": 18, "tool_time_ms": 4,
    }
    metrics = calculate_metrics(case, mode("generated_skills"), actual, observations)
    assert metrics["exact_match"] == 1
    assert metrics["answer_accuracy"] == 1
    assert metrics["generated_skill_loaded"] == 1


def test_generated_routing_requires_skill_expected_by_case() -> None:
    case = ready_case()
    case.raw["expected_skills"] = ["generated_headcount"]
    actual = NormalizedAnswer({
        "result": case.raw["evaluation_contract"]["gold_result"],
        "message": None,
    }, None)
    observations = {
        "loaded_skills": ["another_generated_skill"],
        "heimdall_calls": 1,
        "mcp_query_calls": 0,
        "failed_tool_calls": 0,
        "total_tokens": 1,
        "latency_ms": 1,
        "agent_duration_ms": 1,
        "tool_time_ms": 1,
    }

    metrics = calculate_metrics(
        case, mode("generated_skills"), actual, observations,
    )

    assert metrics["generated_skill_loaded"] == 0


def test_generated_routing_is_not_applicable_without_expected_generated_skill() -> None:
    case = ready_case()
    actual = NormalizedAnswer({
        "result": case.raw["evaluation_contract"]["gold_result"],
        "message": None,
    }, None)
    observations = {
        "loaded_skills": ["generated_headcount"],
        "heimdall_calls": 1,
        "mcp_query_calls": 0,
        "failed_tool_calls": 0,
        "total_tokens": 1,
        "latency_ms": 1,
        "agent_duration_ms": 1,
        "tool_time_ms": 1,
    }

    metrics = calculate_metrics(
        case, mode("generated_skills"), actual, observations,
    )

    assert metrics["generated_skill_loaded"] is None


def test_refusal_accuracy_is_outcome_based() -> None:
    case = ready_case()
    case.raw["category"] = "access_control"
    case.raw["evaluation_contract"] = {
        "expected_outcome": "access_control",
        "gold_result": None,
        "comparison": {
            "ordered": False, "row_key": [], "allow_extra_rows": False,
            "numeric_absolute_tolerance": 0,
        },
    }
    observations = {
        "loaded_skills": [], "heimdall_calls": 1, "mcp_query_calls": 1,
        "failed_tool_calls": 1, "total_tokens": 1, "latency_ms": 1,
        "agent_duration_ms": 1, "tool_time_ms": 1,
        "http_statuses": [403], "error_codes": ["forbidden"],
        "mcp_query_rows": [],
    }
    metrics = calculate_metrics(
        case, mode(),
        NormalizedAnswer(
            {"result": None, "message": "Нет доступа"},
            None,
        ),
        observations,
    )
    assert metrics["outcome_accuracy"] == 1
    assert metrics["correct_refusal"] == 1
    assert metrics["answer_accuracy"] == 1


def test_exact_match_is_not_limited_to_tabular_results() -> None:
    case = ready_case()
    case.raw["response_contract"]["result_schema"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["employee_count"],
        "properties": {"employee_count": {"type": "number"}},
    }
    case.raw["evaluation_contract"] = {
        "expected_outcome": "answer",
        "gold_result": {"employee_count": 10},
        "comparison": {
            "ordered": False, "row_key": [], "allow_extra_rows": False,
            "numeric_absolute_tolerance": 0.5,
        },
    }
    observations = {
        "loaded_skills": [], "heimdall_calls": 1, "mcp_query_calls": 1,
        "failed_tool_calls": 0, "total_tokens": 1, "latency_ms": 1,
        "agent_duration_ms": 1, "tool_time_ms": 1,
    }
    metrics = calculate_metrics(
        case, mode(),
        NormalizedAnswer({
            "result": {"employee_count": 10.4},
            "message": None,
        }, None),
        observations,
    )
    assert metrics["exact_match"] == 1


def test_no_data_outcome_is_inferred_from_trace_not_agent_label() -> None:
    case = ready_case()
    case.raw["category"] = "no_data"
    case.raw["evaluation_contract"] = {
        "expected_outcome": "no_data",
        "gold_result": None,
        "comparison": {
            "ordered": False, "row_key": [], "allow_extra_rows": False,
            "numeric_absolute_tolerance": 0,
        },
    }
    actual = NormalizedAnswer({"result": None, "message": "Данные не найдены"}, None)
    observations = {
        "loaded_skills": [], "heimdall_calls": 1, "mcp_query_calls": 1,
        "failed_tool_calls": 0, "total_tokens": 1, "latency_ms": 1,
        "agent_duration_ms": 1, "tool_time_ms": 1, "http_statuses": [200],
        "error_codes": [], "mcp_query_rows": [],
    }
    without_evidence = calculate_metrics(case, mode(), actual, observations)
    assert without_evidence["outcome_accuracy"] == 0
    observations["mcp_query_rows"] = [0]
    with_evidence = calculate_metrics(case, mode(), actual, observations)
    assert with_evidence["outcome_accuracy"] == 1
