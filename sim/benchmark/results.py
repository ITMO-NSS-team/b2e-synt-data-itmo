"""Immutable raw outputs, derived scores and aggregate mode comparison."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class RunResult:
    """One matrix cell after preflight, execution and scoring.

    Attributes:
        run_id: Unique id ``{eval_id}-{case_id}-{mode}-{repetition}``.
        comparison_group_id: Shared id for the same case × repetition across modes.
        case_id: Authorial case id.
        mode: Arm name.
        repetition: 1-based repeat index.
        status: ``completed``, ``normalization_pending``, ``execution_failed``,
            or a preflight skip status.
        response: Raw stand payload plus case snapshot; ``None`` if skipped.
        score: Metrics and normalized answer; ``None`` if skipped.
        trace: Span tree written beside the response, if any.
    """
    run_id: str
    comparison_group_id: str
    case_id: str
    mode: str
    repetition: int
    status: str
    response: dict[str, Any] | None
    score: dict[str, Any] | None
    trace: dict[str, Any] | None


class ResultWriter:
    """Write each raw response once; scores stay independently reproducible."""

    def __init__(self, results_root: str | Path, eval_id: str) -> None:
        """Create an empty evaluation directory.

        Args:
            results_root: Parent directory for all evaluations.
            eval_id: Subdirectory name; must not already contain files.

        Raises:
            ValueError: If ``results_root/eval_id`` already has content.
        """
        self.root = Path(results_root).resolve() / eval_id
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError(f"evaluation output already exists: {self.root}")
        self.traces = self.root / "traces"
        self.responses_path = self.root / "responses.jsonl"
        self.scores_path = self.root / "scores.jsonl"
        self.traces.mkdir(parents=True, exist_ok=True)

    def write(self, result: RunResult) -> None:
        """Append one response and score; write the trace file once.

        Args:
            result: Finished or skipped matrix cell.
        """
        response = dict(result.response or {})
        if result.trace is not None:
            trace_path = self.traces / f"{result.run_id}.json"
            trace_path.write_text(_json(result.trace) + "\n", encoding="utf-8")
            response["trace_path"] = str(trace_path.relative_to(self.root))
        raw_record = {
            "run_id": result.run_id,
            "comparison_group_id": result.comparison_group_id,
            "case_id": result.case_id,
            "mode": result.mode,
            "repetition": result.repetition,
            "status": result.status,
            **response,
        }
        _append_jsonl(self.responses_path, raw_record)
        if result.score is not None:
            _append_jsonl(self.scores_path, result.score)

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        """Persist the exact requested and live conditions before any run.

        Args:
            manifest: Secret-free experiment description.

        Returns:
            Path of ``run-manifest.json``.

        Raises:
            ValueError: If a manifest was already written in this directory.
        """
        path = self.root / "run-manifest.json"
        if path.exists():
            raise ValueError(f"run manifest already exists: {path}")
        path.write_text(_json(manifest) + "\n", encoding="utf-8")
        return path

    def write_summary(self, results: Iterable[RunResult]) -> Path:
        """Write aggregate per-mode metrics and pairwise deltas.

        Args:
            results: All cells from this evaluation.

        Returns:
            Path of ``summary.json``.
        """
        summary = summarize_results(results)
        path = self.root / "summary.json"
        path.write_text(_json(summary) + "\n", encoding="utf-8")
        return path


def summarize_results(results: Iterable[RunResult]) -> dict[str, Any]:
    """Aggregate completed runs by mode and compare overlapping arms.

    Args:
        results: Cells from one evaluation, including skips.

    Returns:
        Mapping with ``n_runs``, skip counts, ``by_mode`` means and
        ``comparisons`` (delta / relative change). Only ``completed`` rows
        with a score enter the means.
    """
    rows = list(results)
    complete = [row for row in rows if row.status == "completed" and row.score]
    modes = sorted({row.mode for row in rows})
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        group = [row for row in complete if row.mode == mode]
        skipped = [row for row in rows if row.mode == mode and row.status != "completed"]
        metrics = [row.score["metrics"] for row in group]
        by_mode[mode] = {
            "n_runs": len(group),
            "n_skipped": len(skipped),
            **{
                name: _mean(metric.get(name) for metric in metrics)
                for name in (
                    "answer_accuracy", "exact_match", "outcome_accuracy",
                    "correct_refusal", "generated_skill_loaded",
                    "heimdall_calls", "mcp_query_calls", "failed_tool_calls",
                    "total_tokens", "latency_ms", "agent_duration_ms", "tool_time_ms",
                )
            },
        }
    comparisons = {}
    pairs = (
        ("existing_skills", "skills_disabled"),
        ("generated_skills", "skills_disabled"),
        ("generated_skills", "existing_skills"),
    )
    for target, baseline in pairs:
        if target not in by_mode or baseline not in by_mode:
            continue
        key = f"{target}_vs_{baseline}"
        comparisons[key] = _compare(by_mode[target], by_mode[baseline])
    return {
        "n_runs": len(complete),
        "n_normalization_pending": sum(row.status == "normalization_pending" for row in rows),
        "n_skipped": len(rows) - len(complete),
        "by_mode": by_mode,
        "comparisons": comparisons,
    }


def _compare(target: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in (
        "answer_accuracy", "exact_match", "outcome_accuracy", "correct_refusal",
        "generated_skill_loaded", "heimdall_calls", "mcp_query_calls",
        "failed_tool_calls", "total_tokens", "latency_ms",
    ):
        a, b = target.get(name), baseline.get(name)
        if a is None or b is None:
            output[name] = {"delta": None, "relative_change": None}
            continue
        output[name] = {
            "delta": a - b,
            "relative_change": ((a - b) / b) if b != 0 else None,
        }
        if name == "answer_accuracy":
            output[name]["delta_pp"] = 100.0 * (a - b)
            output[name]["error_reduction"] = (
                ((1.0 - b) - (1.0 - a)) / (1.0 - b) if b != 1.0 else None
            )
    return output


def _mean(values: Iterable[Any]) -> float | None:
    kept = [
        float(value) for value in values
        if value is not None and not isinstance(value, bool)
    ]
    return sum(kept) / len(kept) if kept else None


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json(value) + "\n")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
