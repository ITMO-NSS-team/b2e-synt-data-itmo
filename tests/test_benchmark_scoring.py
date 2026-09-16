"""Unit tests for deterministic answer normalization and metrics."""
from __future__ import annotations

import json
from pathlib import Path

from sim.benchmark.cases import BenchmarkCase
from sim.benchmark.modes import SKILL_TOOLS, BenchmarkMode, CommonConditions, ModeConfig
from sim.benchmark.scoring import NormalizedAnswer, calculate_metrics, normalize_answer

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "benchmarking/cases/case-0001.json"


def ready_case() -> BenchmarkCase:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update({
        "case_id": "case-ready", "status": "ready", "employee_id": "123",
        "employee_role": "manager", "snapshot_id": "heimdall-sandbox@test",
        "gold_answer": {
            "outcome": "answer",
            "rows": [
                {"grade": 10, "employee_count": 3},
                {"grade": 11, "employee_count": 2},
            ],
        },
    })
    return BenchmarkCase(Path("/authorial/case-ready.json"), raw)


def mode(name: str = "heimdall_skills") -> ModeConfig:
    common = CommonConditions(
        "model", 0.0, "prompt@1", "heimdall-sandbox@test", True, "instant", ()
    )
    enabled = name != BenchmarkMode.SKILLS_DISABLED
    generated_names = ("generated_headcount",) if name == BenchmarkMode.GENERATED_SKILL else ()
    return ModeConfig(
        name, SKILL_TOOLS if enabled else DATA_TOOLS, None, None, None, None,
        generated_names, common, skills_enabled=enabled,
    )


def test_normalizer_extracts_fenced_json_and_rejects_prose() -> None:
    case = ready_case()
    fence = "\x60\x60\x60"
    answer = (
        f"Результат:\n{fence}json\n"
        + json.dumps(case.raw["gold_answer"], ensure_ascii=False)
        + f"\n{fence}"
    )
    normalized = normalize_answer(answer, case.raw["gold_contract"])
    assert normalized.value == case.raw["gold_answer"]
    assert normalized.error is None
    missing = normalize_answer("Получилось пять сотрудников.", case.raw["gold_contract"])
    assert missing.value is None
    assert "no JSON object" in missing.error


def test_metrics_respect_order_tolerance_and_generated_routing() -> None:
    case = ready_case()
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
    metrics = calculate_metrics(case, mode("generated_skill"), actual, observations)
    assert metrics["exact_match"] == 1
    assert metrics["answer_accuracy"] == 1
    assert metrics["generated_skill_loaded"] == 1


def test_refusal_accuracy_is_outcome_based() -> None:
    case = ready_case()
    case.raw["category"] = "access_control"
    case.raw["gold_answer"] = {"outcome": "access_control", "rows": []}
    observations = {
        "loaded_skills": [], "heimdall_calls": 1, "mcp_query_calls": 1,
        "failed_tool_calls": 1, "total_tokens": 1, "latency_ms": 1,
        "agent_duration_ms": 1, "tool_time_ms": 1,
    }
    metrics = calculate_metrics(
        case, mode(),
        NormalizedAnswer({"outcome": "access_control", "rows": []}, None),
        observations,
    )
    assert metrics["outcome_accuracy"] == 1
    assert metrics["correct_refusal"] == 1
    assert metrics["answer_accuracy"] == 1
