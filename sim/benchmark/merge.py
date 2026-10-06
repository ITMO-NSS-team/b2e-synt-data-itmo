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
) -> tuple[list[RunResult], ResultWriter, bool]:
    """Combine disjoint mode phases that share one logical evaluation id."""
    phases = [Path(path).resolve() for path in phase_dirs]
    if len(phases) < 2:
        raise ValueError("at least two benchmark phase directories are required")
    manifests = [_read_json(path / "run-manifest.json") for path in phases]
    _validate_manifests(manifests, eval_id)

    writer = ResultWriter(results_root, eval_id)
    combined_manifest = dict(manifests[0])
    combined_manifest["schema_version"] = "1.1"
    combined_manifest["modes"] = [
        mode
        for manifest in manifests
        for mode in manifest["modes"]
    ]
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
    writer.write_summary(results)
    return results, writer, False


def _validate_manifests(manifests: list[dict[str, Any]], eval_id: str) -> None:
    first = manifests[0]
    stable = ("eval_id", "check_only", "cases_path", "case_ids", "repetitions")
    if first.get("eval_id") != eval_id:
        raise ValueError("benchmark phase eval_id does not match merge eval_id")
    seen_modes: set[str] = set()
    for manifest in manifests:
        for field in stable:
            if manifest.get(field) != first.get(field):
                raise ValueError(f"benchmark phases differ by {field}")
        for mode in manifest.get("modes", []):
            name = mode.get("name")
            if not name or name in seen_modes:
                raise ValueError(f"duplicate or empty mode across benchmark phases: {name}")
            seen_modes.add(name)


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
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        results, writer, check_only = merge_phases(
            args.phase, results_root=args.results, eval_id=args.eval_id,
        )
        if check_only:
            report = {
                "results_dir": str(writer.root),
                "preflight": str(writer.root / "preflight.json"),
                "agent_calls": 0,
            }
        else:
            summary = summarize_results(results)
            report = {
                "results_dir": str(writer.root),
                "runs": len(results),
                "n_completed": summary["n_completed"],
                "n_scored": summary["n_scored"],
                "n_review": len(summary["review"]),
                "n_normalization_pending": summary["n_normalization_pending"],
                "n_condition_invalid": summary["n_condition_invalid"],
                "n_unscored": summary["n_unscored"],
                "summary": str(writer.root / "summary.json"),
            }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"benchmark phase merge failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
