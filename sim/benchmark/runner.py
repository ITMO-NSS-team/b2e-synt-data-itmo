"""Orchestrate preflight, isolated turns, normalization, scoring and output."""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .cases import BenchmarkCase
from .contracts import (
    PROMPT_RENDERER_VERSION, render_agent_query, response_contract_hash,
)
from .execution import (
    AgentRequest, AgentTurn, ModeActivator, SessionExecutor,
    fatal_turn_error, trace_observations,
)
from .modes import ModeConfig, ModeConfigs
from .preflight import PreflightResult
from .results import ResultWriter, RunResult
from .scoring import calculate_metrics, normalize_answer
from sim.fingerprint import RunFingerprint


@dataclass(frozen=True, slots=True)
class BenchmarkRunner:
    """Run the selected matrix: preflight, one isolated turn, score, write.

    Attributes:
        eval_id: Prefix for run ids and the results directory name.
        preflight: Checks one case × mode; must not start an LLM session.
        activator: Swaps live agent config around each turn.
        executor: Opens a fresh stand session per sample.
        writer: Optional sink for responses, scores, traces and summary.
        scorer_version: Label stored on every score record.
    """
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
        """Execute every case × mode × repetition in a fixed order.

        Args:
            cases: Ready cases.
            modes: ``ModeConfigs``, a name→config mapping, or an iterable of configs.
            repetitions: Independent repeats per cell; must be ``>= 1``.

        Returns:
            One ``RunResult`` per cell, including skipped preflight statuses.

        Raises:
            ValueError: If ``repetitions`` is invalid or mode names collide.
        """
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
        """Run, skip or fail a single matrix cell.

        Args:
            case: Authorial case.
            mode: Arm to activate.
            repetition: 1-based repeat index.

        Returns:
            Result with a preflight skip status, ``completed``,
            ``normalization_pending`` or ``execution_failed``.
        """
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
            public_contract = case.raw["response_contract"]
            contract_hash = response_contract_hash(public_contract)
            rendered_query = render_agent_query(case.raw["query"], public_contract)
            request = AgentRequest(
                query=rendered_query,
                employee_id=case.raw["employee_id"],
                config_ref=activated.config_ref,
                metadata={
                    "eval_id": self.eval_id,
                    "case_id": case.case_id,
                    "mode": mode.name,
                    "run_id": run_id,
                    "comparison_group_id": group_id,
                    "response_contract_hash": contract_hash,
                    "prompt_renderer_version": PROMPT_RENDERER_VERSION,
                },
            )
            try:
                turn = self.executor.execute(request)
            except Exception:
                turn = AgentTurn(answer="", error=traceback.format_exc())
        finally:
            self.activator.deactivate(mode)

        fatal_error = fatal_turn_error(turn.error, answer=turn.answer)
        fingerprint_error = None if fatal_error else _fingerprint_error(checked, turn)
        effective_error = fatal_error or fingerprint_error
        observations = trace_observations(turn)
        normalized = normalize_answer(turn.answer, case.raw["response_contract"])
        metrics = calculate_metrics(case, mode, normalized, observations)
        if effective_error:
            metrics["answer_accuracy"] = 0
            metrics["exact_match"] = 0
            metrics["outcome_accuracy"] = 0
            if metrics["correct_refusal"] is not None:
                metrics["correct_refusal"] = 0
        status = (
            "execution_failed" if effective_error else
            "normalization_pending" if normalized.value is None else
            "completed"
        )
        response = {
            "condition_id": checked.fingerprint.condition_id if checked.fingerprint else None,
            "catalog_hash": activated.catalog_hash,
            "case_snapshot": {
                "category": case.raw["category"],
                "original_query": case.raw["query"],
                "rendered_query": rendered_query,
                "employee_role": case.raw["employee_role"],
                "employee_id": case.raw["employee_id"],
                "expected_skills": case.raw["expected_skills"],
                "response_contract": case.raw["response_contract"],
                "response_contract_hash": contract_hash,
                "prompt_renderer_version": PROMPT_RENDERER_VERSION,
                "evaluation_contract": case.raw["evaluation_contract"],
            },
            "response": {"raw_answer": turn.answer, "error": effective_error},
            "observations": observations,
            "session_id": turn.session_id,
            "trace_id": turn.trace_id,
            "live_fingerprint": turn.fingerprint,
        }
        reasons = []
        if effective_error:
            reasons.append(effective_error)
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


def _fingerprint_error(
    checked: PreflightResult, turn: AgentTurn,
) -> str | None:
    """Refuse results produced under conditions other than preflight approved.

    Args:
        checked: Preflight result that issued the expected fingerprint.
        turn: Live stand turn.

    Returns:
        Error string if fingerprints differ or are missing; ``None`` if the
        turn already failed or fingerprints match.
    """
    if fatal_turn_error(turn.error, answer=turn.answer):
        return None
    if checked.fingerprint is None:
        return "preflight returned no experiment fingerprint"
    if turn.fingerprint is None:
        return "stand session returned no experiment fingerprint"
    try:
        live = RunFingerprint.create(**turn.fingerprint)
    except Exception as exc:
        return f"invalid stand fingerprint: {exc}"
    if live != checked.fingerprint:
        expected = checked.fingerprint.as_dict()
        actual = live.as_dict()
        changed = sorted(key for key in expected if expected[key] != actual[key])
        return "stand fingerprint differs from preflight: " + ", ".join(changed)
    return None
