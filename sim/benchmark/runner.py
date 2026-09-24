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
from .scoring import NormalizedAnswer, calculate_metrics, normalize_answer
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
        """Run one matrix cell, or skip it before any LLM call.

        Args:
            case: Authorial case.
            mode: Arm to activate.
            repetition: 1-based repeat index.

        Returns:
            Result with a preflight skip status, ``completed``,
            ``normalization_pending``, ``condition_invalid`` or ``unscored``.
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
                    # Denied MCP attempts are reported by the harness, but they
                    # never reach Heimdall and therefore produce no bridge span.
                    # Trace completeness follows the selected arm's tool surface.
                    "heimdall_access": (
                        "enabled" if mode.tool_subset else "disabled"
                    ),
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

        observations = trace_observations(turn)
        normalized = normalize_answer(turn.answer, case.raw["response_contract"])
        status, review_reason = _classify_turn(checked, turn, normalized)
        metrics = calculate_metrics(case, mode, normalized, observations)
        if status != "completed":
            for name in (
                "answer_accuracy", "exact_match", "outcome_accuracy", "correct_refusal",
            ):
                metrics[name] = None
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
            "response": {"raw_answer": turn.answer, "error": review_reason},
            "observations": observations,
            "session_id": turn.session_id,
            "trace_id": turn.trace_id,
            "live_fingerprint": turn.fingerprint,
        }
        reasons = []
        if review_reason:
            reasons.append(review_reason)
        if normalized.error and status == "normalization_pending":
            if normalized.error not in reasons:
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


def _classify_turn(
    checked: PreflightResult, turn: AgentTurn, normalized: NormalizedAnswer,
) -> tuple[str, str | None]:
    """Label a finished stand turn without aborting the rest of the matrix.

    Args:
        checked: Preflight result that issued the expected fingerprint.
        turn: Live stand turn.
        normalized: Extracted JSON, or a schema/extraction failure.

    Returns:
        Status and a review reason. ``completed`` has a ``None`` reason.
    """
    fatal_error = fatal_turn_error(turn.error, answer=turn.answer)
    if fatal_error:
        return "unscored", fatal_error
    fingerprint_error = _fingerprint_error(checked, turn)
    if fingerprint_error:
        return "condition_invalid", fingerprint_error
    if turn.trace is None:
        session = turn.session_id or "unknown"
        return "unscored", f"trace unavailable for session {session}"
    if normalized.value is None:
        return "normalization_pending", normalized.error
    return "completed", None


def _fingerprint_error(
    checked: PreflightResult, turn: AgentTurn,
) -> str | None:
    """Detect live conditions that drifted from preflight.

    Args:
        checked: Preflight result that issued the expected fingerprint.
        turn: Live stand turn.

    Returns:
        Review reason if fingerprints differ or are missing; ``None`` if they
        match. Does not abort the evaluation.
    """
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
