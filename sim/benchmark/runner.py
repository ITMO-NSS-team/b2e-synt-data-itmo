"""Orchestrate preflight, isolated turns, normalization, scoring and output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .cases import BenchmarkCase
from .execution import (
    AgentRequest, AgentTurn, ModeActivator, SessionExecutor, trace_observations,
)
from .modes import ModeConfig, ModeConfigs
from .preflight import PreflightResult
from .results import ResultWriter, RunResult
from .scoring import calculate_metrics, normalize_answer


@dataclass(frozen=True, slots=True)
class BenchmarkRunner:
    eval_id: str
    preflight: Callable[[BenchmarkCase, ModeConfig], PreflightResult]
    activator: ModeActivator
    executor: SessionExecutor
    writer: ResultWriter | None = None
    scorer_version: str = "benchmark-scorer@1"

    def run(
        self,
        cases: Iterable[BenchmarkCase],
        modes: ModeConfigs | Mapping[str, ModeConfig] | Iterable[ModeConfig],
        *,
        repetitions: int = 1,
    ) -> list[RunResult]:
        if repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        selected_modes = _mode_list(modes)
        results: list[RunResult] = []
        for case in cases:
            for repetition in range(1, repetitions + 1):
                for mode in selected_modes:
                    result = self.run_one(case, mode, repetition)
                    results.append(result)
                    if self.writer is not None:
                        self.writer.write(result)
        if self.writer is not None:
            self.writer.write_summary(results)
        return results

    def run_one(
        self, case: BenchmarkCase, mode: ModeConfig, repetition: int,
    ) -> RunResult:
        run_id = f"{self.eval_id}-{case.case_id}-{mode.name}-{repetition:02d}"
        group_id = f"{self.eval_id}-{case.case_id}-{repetition:02d}"
        checked = self.preflight(case, mode)
        if checked.status != "ready":
            return RunResult(
                run_id, group_id, case.case_id, mode.name, repetition,
                checked.status, None, None, None,
            )

        activated = self.activator.activate(mode)
        try:
            request = AgentRequest(
                query=case.raw["query"],
                employee_id=case.raw["employee_id"],
                config_ref=activated.config_ref,
                metadata={
                    "eval_id": self.eval_id,
                    "case_id": case.case_id,
                    "mode": mode.name,
                    "run_id": run_id,
                    "comparison_group_id": group_id,
                },
            )
            try:
                turn = self.executor.execute(request)
            except Exception as exc:
                turn = AgentTurn(answer="", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.activator.deactivate(mode)

        observations = trace_observations(turn)
        normalized = normalize_answer(turn.answer, case.raw["gold_contract"])
        metrics = calculate_metrics(case, mode, normalized, observations)
        if turn.error:
            metrics["answer_accuracy"] = 0
            metrics["exact_match"] = 0
            metrics["outcome_accuracy"] = 0
            if metrics["correct_refusal"] is not None:
                metrics["correct_refusal"] = 0
        status = "normalization_pending" if normalized.value is None and not turn.error else "completed"
        response = {
            "condition_id": checked.fingerprint.condition_id if checked.fingerprint else None,
            "catalog_hash": activated.catalog_hash,
            "case_snapshot": {
                "category": case.raw["category"],
                "query": case.raw["query"],
                "employee_role": case.raw["employee_role"],
                "employee_id": case.raw["employee_id"],
                "expected_outcome": case.raw["gold_answer"]["outcome"],
                "expected_skills": case.raw["expected_skills"],
                "gold_answer": case.raw["gold_answer"],
                "gold_contract": case.raw["gold_contract"],
                "gold_comparison": case.raw["gold_comparison"],
            },
            "response": {"raw_answer": turn.answer, "error": turn.error},
            "observations": observations,
            "session_id": turn.session_id,
            "trace_id": turn.trace_id,
        }
        reasons = []
        if turn.error:
            reasons.append(turn.error)
        if normalized.error:
            reasons.append(normalized.error)
        score = {
            "run_id": run_id,
            "scorer_version": self.scorer_version,
            "normalized_answer": normalized.value,
            "metrics": metrics,
            "llm_judge": {
                "applicable": False,
                "pass": None,
                "scores": None,
                "explanation": None,
            },
            "reasons": reasons,
        }
        return RunResult(
            run_id, group_id, case.case_id, mode.name, repetition,
            status, response, score, turn.trace,
        )


def _mode_list(
    modes: ModeConfigs | Mapping[str, ModeConfig] | Iterable[ModeConfig],
) -> list[ModeConfig]:
    values = list(modes.values()) if isinstance(modes, Mapping) else list(modes)
    names = [mode.name for mode in values]
    if len(names) != len(set(names)):
        raise ValueError("benchmark modes contain duplicate names")
    return sorted(values, key=lambda mode: mode.name)
