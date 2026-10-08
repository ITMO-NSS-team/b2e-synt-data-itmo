"""Merge catalog-isolated benchmark phases into one evaluation artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .results import ResultWriter, RunResult, summarize_results


def merge_phases(
    phase_dirs: Iterable[str | Path],
    *,
    results_root: str | Path,
    eval_id: str,
    generated_plan: str | Path | None = None,
) -> tuple[list[RunResult], ResultWriter, bool]:
    """Combine disjoint mode phases that share one logical evaluation id."""
    phases = [Path(path).resolve() for path in phase_dirs]
    if not phases:
        raise ValueError("at least one benchmark phase directory is required")
    manifests = [_read_json(path / "run-manifest.json") for path in phases]
    _validate_manifests(manifests, eval_id)
    plan = _read_json(Path(generated_plan)) if generated_plan else None
    _validate_generated_plan_execution(manifests, plan)

    writer = ResultWriter(results_root, eval_id)
    combined_manifest = dict(manifests[0])
    combined_manifest["schema_version"] = "1.1"
    combined_manifest["modes"] = _unique_modes(manifests)
    combined_manifest["case_ids"] = sorted({
        case_id for manifest in manifests for case_id in manifest["case_ids"]
    })
    if plan:
        combined_manifest["cases_path"] = plan["cases_source"]
        combined_manifest["generated_plan"] = plan
    combined_manifest["catalog_phases"] = [
        {
            "modes": [mode["name"] for mode in manifest["modes"]],
            "live_stand": manifest["live_stand"],
        }
        for manifest in manifests
    ]
    writer.write_manifest(combined_manifest)

    check_only = bool(manifests[0]["check_only"])
    if check_only:
        report = [
            row
            for phase in phases
            for row in _read_json(phase / "preflight.json")
        ]
        report.sort(key=lambda row: (row["case_id"], row["mode"]))
        (writer.root / "preflight.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [], writer, True

    results: list[RunResult] = []
    seen_run_ids: set[str] = set()
    for phase in phases:
        scores = {
            row["run_id"]: row
            for row in _read_jsonl(phase / "scores.jsonl")
        }
        for raw in _read_jsonl(phase / "responses.jsonl"):
            run_id = raw["run_id"]
            if run_id in seen_run_ids:
                raise ValueError(f"duplicate run_id across benchmark phases: {run_id}")
            seen_run_ids.add(run_id)
            trace = None
            if trace_path := raw.get("trace_path"):
                trace = _read_json(phase / trace_path)
            response = {
                key: value
                for key, value in raw.items()
                if key not in {
                    "run_id", "comparison_group_id", "case_id", "mode",
                    "repetition", "status", "trace_path",
                }
            }
            results.append(RunResult(
                run_id=run_id,
                comparison_group_id=raw["comparison_group_id"],
                case_id=raw["case_id"],
                mode=raw["mode"],
                repetition=int(raw["repetition"]),
                status=raw["status"],
                response=response,
                score=scores.get(run_id),
                trace=trace,
            ))

    results.sort(key=lambda row: (row.case_id, row.repetition, row.mode))
    for result in results:
        writer.write(result)
    writer.write_summary(results, extra=_selection_summary(plan))
    return results, writer, False


def _validate_manifests(manifests: list[dict[str, Any]], eval_id: str) -> None:
    first = manifests[0]
    stable = ("eval_id", "check_only", "repetitions")
    if first.get("eval_id") != eval_id:
        raise ValueError("benchmark phase eval_id does not match merge eval_id")
    seen_cells: set[tuple[str, str]] = set()
    for manifest in manifests:
        for field in stable:
            if manifest.get(field) != first.get(field):
                raise ValueError(f"benchmark phases differ by {field}")
        for mode in manifest.get("modes", []):
            name = mode.get("name")
            if not name:
                raise ValueError("empty mode across benchmark phases")
            for case_id in manifest.get("case_ids", []):
                cell = (case_id, name)
                if cell in seen_cells:
                    raise ValueError(
                        "duplicate case/mode across benchmark phases: "
                        f"{case_id}/{name}"
                    )
                seen_cells.add(cell)


def _unique_modes(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one top-level entry per mode; phase-specific catalogs stay below."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        for mode in manifest["modes"]:
            if mode["name"] not in seen:
                result.append(mode)
                seen.add(mode["name"])
    return result


def _validate_generated_plan_execution(
    manifests: list[dict[str, Any]], plan: dict[str, Any] | None,
) -> None:
    """Refuse a partial merge when a planned generated variant is absent."""
    if not plan:
        return
    actual = {
        mode["name"]
        for manifest in manifests
        for mode in manifest.get("modes", [])
    }
    expected = {
        f"generated_skills@{group['variant_id']}"
        for group in plan.get("groups", [])
    }
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            "generated benchmark phases are incomplete; missing variants: "
            + ", ".join(missing)
        )


def _selection_summary(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Expose dataset coverage without turning excluded cases into run cells."""
    if not plan:
        return {}
    source = plan.get("source_case_ids", [])
    eligible = plan.get("eligible_case_ids", [])
    selected = plan.get("selected_case_ids", [])
    excluded = plan.get("excluded", [])
    partially_covered = plan.get("partially_covered", [])
    reasons = {
        status: sum(row.get("status") == status for row in excluded)
        for status in sorted({row.get("status") for row in excluded})
    }
    return {
        "selection": {
            "n_source_cases": len(source),
            "n_eligible_cases": len(eligible),
            "n_selected_cases": len(selected),
            "n_excluded_cases": len(excluded),
            "n_partially_covered_cases": len(partially_covered),
            "coverage_rate": len(eligible) / len(source) if source else None,
            "excluded_by_reason": reasons,
        }
    }


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing benchmark phase artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Merge catalog-isolated benchmark phases")
    result.add_argument("--phase", action="append", required=True)
    result.add_argument("--results", required=True)
    result.add_argument("--eval-id", required=True)
    result.add_argument("--generated-plan")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        results, writer, check_only = merge_phases(
            args.phase, results_root=args.results, eval_id=args.eval_id,
            generated_plan=args.generated_plan,
        )
        if check_only:
            report = {
                "results_dir": str(writer.root),
                "preflight": str(writer.root / "preflight.json"),
                "agent_calls": 0,
            }
        else:
            summary = json.loads((writer.root / "summary.json").read_text(encoding="utf-8"))
            report = {
                "results_dir": str(writer.root),
                "runs": len(results),
                "n_completed": summary["n_completed"],
                "n_scored": summary["n_scored"],
                "n_review": len(summary["review"]),
                "n_normalization_pending": summary["n_normalization_pending"],
                "n_condition_invalid": summary["n_condition_invalid"],
                "n_unscored": summary["n_unscored"],
                "selection": summary.get("selection"),
                "summary": str(writer.root / "summary.json"),
            }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"benchmark phase merge failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
