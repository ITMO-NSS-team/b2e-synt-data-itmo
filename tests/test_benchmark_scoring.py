"""Unit tests for deterministic answer normalization and metrics."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sim.benchmark.cases import BenchmarkCase
from sim.benchmark.modes import (
    GENERATED_SKILL_TOOLS, GENERAL_KNOWLEDGE_TOOLS, SKILL_TOOLS,
    BenchmarkMode, CommonConditions, ModeConfig,
)
from sim.benchmark.scoring import NormalizedAnswer, calculate_metrics, normalize_answer
from tests.fixtures.constants import EXAMPLE


def ready_case() -> BenchmarkCase:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update({
        "case_id": "case-9999-00", "status": "verified", "employee_id": 123,
        "employee_role": "manager", "snapshot_id": "heimdall-sandbox@test",
        "gold_answer": {
            "outcome": "answer",
            "rows": [
                {"grade": 10, "employee_count": 3},
                {"grade": 11, "employee_count": 2},
            ],
        },
    })
    return BenchmarkCase(Path("/authorial/case-9999-00.json"), raw)


def mode(
    name: str = "existing_skills",
    generated_names: tuple[str, ...] = ("generated_headcount",),
) -> ModeConfig:
    common = CommonConditions(
        "model", 0.0, "prompt@1", "heimdall-sandbox@test", True, "instant", ()
    )
    enabled = name != BenchmarkMode.GENERAL_KNOWLEDGE
    generated_names = (
        generated_names
        if name in (
            BenchmarkMode.GENERATED_SKILLS,
            BenchmarkMode.EXISTING_PLUS_GENERATED,
        )
        else ()
    )
    tools = (
        GENERATED_SKILL_TOOLS
        if name == BenchmarkMode.GENERATED_SKILLS
        else SKILL_TOOLS
    )
    return ModeConfig(
        name, tools if enabled else GENERAL_KNOWLEDGE_TOOLS,
        None, None, None, None,
        generated_names, common, skills_enabled=enabled,
    )


def test_normalizer_extracts_fenced_json_and_rejects_prose() -> None:
    case = ready_case()
    fence = "\x60\x60\x60"
    expected = case.raw["gold_answer"]
    answer = (
        f"Результат:\n{fence}json\n"
        + json.dumps(expected, ensure_ascii=False)
        + f"\n{fence}"
    )
    normalized = normalize_answer(answer, case.raw["gold_contract"])
    assert normalized.value == expected
    assert normalized.error is None
    missing = normalize_answer("Получилось пять сотрудников.", case.raw["gold_contract"])
    assert missing.value is None
    assert "no JSON object" in missing.error


def test_metrics_respect_order_tolerance_and_generated_routing() -> None:
    case = ready_case()
    case.raw["expected_skills"] = ["generated_headcount"]
    case.raw["gold_comparison"]["numeric_absolute_tolerance"] = 1
    actual = NormalizedAnswer({
        "outcome": "answer",
        "rows": [
            {"grade": 11, "employee_count": 3},
            {"grade": 10, "employee_count": 2},
        ],
    }, None)
    observations = {
        "loaded_skills": ["generated_headcount"], "heimdall_calls": 3,
        "mcp_query_calls": 1, "failed_tool_calls": 0, "total_tokens": 10,
        "latency_ms": 20, "agent_duration_ms": 18, "tool_time_ms": 4,
    }
    metrics = calculate_metrics(case, mode("generated_skills"), actual, observations)
    assert metrics["exact_match"] == 1
    assert metrics["answer_accuracy"] == 1
    assert metrics["any_skill_loaded"] == 1
    assert metrics["expected_skill_loaded"] == 1
    assert metrics["generated_skill_loaded"] == 1


def test_generated_routing_requires_skill_expected_by_case() -> None:
    case = ready_case()
    case.raw["expected_skills"] = ["generated_headcount"]
    actual = NormalizedAnswer(case.raw["gold_answer"], None)
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
    assert metrics["any_skill_loaded"] == 1
    assert metrics["expected_skill_loaded"] == 0


def test_existing_skill_load_matches_catalog_version_only(tmp_path: Path) -> None:
    (tmp_path / "compare_people.md").write_text(
        "---\nname: compare_people\ntitle: Compare people\nkind: reference\n"
        "domain: general\ndescription: Compare employees\nversion: 1.0.0\n"
        "status: active\n---\n\nCompare two people.\n",
        encoding="utf-8",
    )
    case = ready_case()
    actual = NormalizedAnswer(case.raw["gold_answer"], None)
    observations = {"loaded_skills": ["compare_people"]}
    existing = replace(mode(), catalog_path=str(tmp_path))

    case.raw["expected_skills"] = ["compare_people-v1-0-0"]
    assert calculate_metrics(
        case, existing, actual, observations,
    )["expected_skill_loaded"] == 1

    case.raw["expected_skills"] = ["compare_people-v1-0-1"]
    assert calculate_metrics(
        case, existing, actual, observations,
    )["expected_skill_loaded"] == 0

    versioned_dir = tmp_path / "versioned"
    versioned_dir.mkdir()
    (versioned_dir / "skill.md").write_text(
        "---\nname: compare_people-v1-0-1\ntitle: Compare people\nkind: reference\n"
        "domain: general\ndescription: Compare employees\nversion: 1.0.0\n"
        "status: active\n---\n\nCompare two people.\n",
        encoding="utf-8",
    )
    observations["loaded_skills"] = ["compare_people-v1-0-1"]
    assert calculate_metrics(
        case, replace(existing, catalog_path=str(versioned_dir)), actual, observations,
    )["expected_skill_loaded"] == 0


def test_combined_mode_scores_only_the_generated_target() -> None:
    case = ready_case()
    case.raw["expected_skills"] = ["generated_headcount"]
    actual = NormalizedAnswer(case.raw["gold_answer"], None)
    observations = {
        "loaded_skills": ["existing_headcount"],
        "heimdall_calls": 1,
        "mcp_query_calls": 0,
        "failed_tool_calls": 0,
        "total_tokens": 1,
        "latency_ms": 1,
        "agent_duration_ms": 1,
        "tool_time_ms": 1,
    }

    not_loaded = calculate_metrics(
        case,
        mode("existing_plus_generated", ("existing_headcount", "generated_headcount")),
        actual,
        observations,
    )
    observations["loaded_skills"] = ["generated_headcount"]
    loaded = calculate_metrics(
        case,
        mode("existing_plus_generated", ("existing_headcount", "generated_headcount")),
        actual,
        observations,
    )

    assert not_loaded["generated_skill_loaded"] == 0
    assert loaded["generated_skill_loaded"] == 1


def test_generated_routing_is_not_applicable_without_expected_generated_skill() -> None:
    case = ready_case()
    actual = NormalizedAnswer(case.raw["gold_answer"], None)
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


def test_access_refusal_requires_heimdall_403_and_empty_rows() -> None:
    case = ready_case()
    case.raw["category"] = "access_control"
    case.raw["gold_answer"] = {"outcome": "access_control", "rows": []}
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
            {"outcome": "access_control", "rows": [], "comment": "Нет доступа"},
            None,
        ),
        observations,
    )
    assert metrics["outcome_accuracy"] == 1
    assert metrics["correct_refusal"] == 1
    assert metrics["answer_accuracy"] == 1

    no_heimdall = dict(observations, heimdall_calls=0, http_statuses=[])
    metrics = calculate_metrics(
        case, mode("general_knowledge"),
        NormalizedAnswer(
            {"outcome": "access_control", "rows": [], "comment": "Нет инструментов"},
            None,
        ),
        no_heimdall,
    )
    assert metrics["outcome_accuracy"] == 1
    assert metrics["correct_refusal"] == 0
    assert metrics["answer_accuracy"] == 0

    leaked = calculate_metrics(
        case, mode(),
        NormalizedAnswer(
            {"outcome": "access_control", "rows": [{"secret": "value"}]},
            None,
        ),
        observations,
    )
    assert leaked["correct_refusal"] == 0
    assert leaked["answer_accuracy"] == 0


def test_policy_refusal_requires_no_data_query() -> None:
    case = ready_case()
    case.raw["category"] = "access_control"
    case.raw["gold_answer"] = {"outcome": "access_control", "rows": []}
    case.raw["gold_comparison"]["refusal_evidence"] = "policy_refusal"
    observations = {
        "loaded_skills": ["absence_limits"],
        "heimdall_calls": 2,
        "mcp_query_calls": 0,
        "failed_tool_calls": 0,
        "total_tokens": 1,
        "latency_ms": 1,
        "agent_duration_ms": 1,
        "tool_time_ms": 1,
        "http_statuses": [],
        "error_codes": [],
        "mcp_query_rows": [],
    }
    refusal = NormalizedAnswer(
        {"outcome": "access_control", "rows": [], "comment": "Только свои данные"},
        None,
    )

    metrics = calculate_metrics(case, mode(), refusal, observations)
    assert metrics["correct_refusal"] == 1
    assert metrics["answer_accuracy"] == 1

    queried = calculate_metrics(
        case, mode(), refusal, dict(observations, mcp_query_calls=1),
    )
    assert queried["correct_refusal"] == 0
    assert queried["answer_accuracy"] == 0


def test_format_failure_is_scored_as_incorrect() -> None:
    metrics = calculate_metrics(
        ready_case(), mode(), NormalizedAnswer(None, "invalid JSON"),
        {"loaded_skills": [], "heimdall_calls": 1},
    )
    assert metrics["answer_accuracy"] == 0
    assert metrics["exact_match"] == 0
    assert metrics["outcome_accuracy"] == 0


def test_exact_match_is_not_limited_to_tabular_results() -> None:
    case = ready_case()
    case.raw["gold_contract"]["properties"]["rows"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["employee_count"],
        "properties": {"employee_count": {"type": "number"}},
    }
    case.raw["gold_answer"] = {
        "outcome": "answer", "rows": {"employee_count": 10},
    }
    case.raw["gold_comparison"] = {
        "ordered": False, "row_key": [], "allow_extra_rows": False,
        "numeric_absolute_tolerance": 0.5,
    }
    observations = {
        "loaded_skills": [], "heimdall_calls": 1, "mcp_query_calls": 1,
        "failed_tool_calls": 0, "total_tokens": 1, "latency_ms": 1,
        "agent_duration_ms": 1, "tool_time_ms": 1,
    }
    metrics = calculate_metrics(
        case, mode(),
        NormalizedAnswer({
            "outcome": "answer", "rows": {"employee_count": 10.4},
        }, None),
        observations,
    )
    assert metrics["exact_match"] == 1


def test_no_data_outcome_uses_the_v2_agent_label() -> None:
    case = ready_case()
    case.raw["category"] = "no_data"
    case.raw["gold_answer"] = {"outcome": "no_data", "rows": []}
    actual = NormalizedAnswer({"outcome": "no_data", "rows": []}, None)
    observations = {
        "loaded_skills": [], "heimdall_calls": 1, "mcp_query_calls": 1,
        "failed_tool_calls": 0, "total_tokens": 1, "latency_ms": 1,
        "agent_duration_ms": 1, "tool_time_ms": 1, "http_statuses": [200],
        "error_codes": [], "mcp_query_rows": [],
    }
    metrics = calculate_metrics(case, mode(), actual, observations)
    assert metrics["outcome_accuracy"] == 1
