"""Self-contained Plotly report for one benchmark evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median
from string import Template
from typing import Any, Iterable

from plotly.offline import get_plotlyjs


_ASSETS = Path(__file__).with_name("report_assets")


def write_report(result_dir: str | Path, results: Iterable[Any], *, publish: bool = True) -> Path:
    """Write a self-contained report beside the benchmark JSON artifacts."""
    root = Path(result_dir)
    rows = [_result_row(result) for result in results]
    path = _write(root, rows)
    return publish_report(path, root.name) if publish else path


def write_report_from_files(result_dir: str | Path) -> Path:
    """Rebuild a report for an existing evaluation directory."""
    root = Path(result_dir)
    scores = {
        row["run_id"]: row
        for row in _read_jsonl(root / "scores.jsonl")
    }
    rows = []
    for response in _read_jsonl(root / "responses.jsonl"):
        score = scores.get(response["run_id"])
        metrics = dict((score or {}).get("metrics") or {})
        _add_skill_metrics(metrics, response)
        rows.append({
            "run_id": response["run_id"],
            "case_id": response["case_id"],
            "mode": response["mode"],
            "repetition": response["repetition"],
            "status": response["status"],
            "metrics": metrics,
        })
    return publish_report(_write(root, rows), root.name)


def publish_report(path: Path, eval_id: str) -> Path:
    if public_root := (os.getenv("BENCHMARK_REPORT_DIR") or "").strip():
        public_path = Path(public_root) / eval_id / "report.html"
        public_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".report-", dir=public_path.parent)
        os.close(descriptor)
        try:
            shutil.copyfile(path, temporary)
            shutil.copymode(path, temporary)
            os.replace(temporary, public_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


def _result_row(result: Any) -> dict[str, Any]:
    metrics = dict((result.score or {}).get("metrics") or {})
    _add_skill_metrics(metrics, getattr(result, "response", None) or {})
    return {
        "run_id": result.run_id,
        "case_id": result.case_id,
        "mode": result.mode,
        "repetition": result.repetition,
        "status": result.status,
        "metrics": metrics,
    }


def _add_skill_metrics(metrics: dict[str, Any], response: dict[str, Any]) -> None:
    """Backfill skill-use metrics for reports created from older runs."""
    loaded = set((response.get("observations") or {}).get("loaded_skills") or ())
    expected = set((response.get("case_snapshot") or {}).get("expected_skills") or ())
    metrics.setdefault("any_skill_loaded", int(bool(loaded)))
    metrics.setdefault("expected_skill_loaded", int(bool(loaded & expected)))


def _write(root: Path, rows: list[dict[str, Any]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    plan = {}
    if (root / "run-manifest.json").is_file():
        plan = json.loads((root / "run-manifest.json").read_text(encoding="utf-8")).get("generated_plan") or {}
    groups = {
        f"{mode}@{group['variant_id']}": group
        for group in plan.get("groups", [])
        for mode in group.get("modes", plan.get("generated_modes", []))
    }
    points = [_point(row, groups.get(str(row["mode"]))) for row in rows]
    payload = json.dumps(
        {
            "eval_id": root.name,
            "runs": points,
            "aggregates": _aggregate(points),
            "coverage": {
                "source_cases": len(plan.get("source_case_ids", [])),
                "selected_cases": len(plan.get("selected_case_ids", [])),
                "excluded": plan.get("excluded", []),
                "unmatched_artifacts": plan.get("unmatched_artifacts", []),
                "catalogs": [
                    {"source": group["source"],
                     "matched": len(group["matched_case_ids"]),
                     "cases": len(group["case_ids"])}
                    for group in plan.get("groups", [])
                    if group.get("scope") == "catalog"
                ],
            } if plan else None,
            "counts": {
                "runs": len(rows),
                "scored": sum(point["accuracy"] is not None for point in points),
                "completed": sum(row["status"] == "completed" for row in rows),
                "review": sum(row["status"] in {
                    "normalization_pending", "condition_invalid", "unscored",
                } for row in rows),
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = Template(_asset("report.html")).substitute(
        styles=_asset("report.css"),
        plotly=get_plotlyjs().replace("</", "<\\/"),
        payload=payload,
        script=_asset("report.js").replace("</", "<\\/"),
    )
    path = root / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def _point(row: dict[str, Any], group: dict[str, Any] | None = None) -> dict[str, Any]:
    mode, method, detail = _mode_parts(str(row["mode"]))
    variant = str(row["mode"]).partition("@")[2] or "—"
    source = "—"
    skill = "—"
    if group:
        if source_path := group.get("source_path"):
            directories = (Path(source_path).parts if group.get("scope") == "catalog"
                           else Path(source_path).parent.parts)
            method = next((part for part in reversed(directories)
                           if part in {"no-context", "compact", "full"}), "—")
            detail = next((part for part in reversed(directories)
                           if part in {"abstract", "medium", "detailed"}), "—")
        source = group.get("source", "—")
        variant = group["variant_id"]
        names = group.get("skill_names", [])
        skill = f"{len(names)} skills" if group.get("scope") == "catalog" else ", ".join(names)
    metrics = row["metrics"]
    latency = metrics.get("latency_ms")
    return {
        "run_id": row["run_id"],
        "case_id": row["case_id"],
        "mode": mode,
        "source": source,
        "variant": variant,
        "skill": skill,
        "method": method,
        "detail": detail,
        "repetition": row["repetition"],
        "status": row["status"],
        "accuracy": metrics.get("answer_accuracy"),
        "format_valid": metrics.get("format_valid"),
        "llm_judge_accuracy": metrics.get("llm_judge_accuracy"),
        "any_skill_loaded": metrics.get("any_skill_loaded"),
        "expected_skill_loaded": metrics.get("expected_skill_loaded"),
        "generated_skill_loaded": metrics.get("generated_skill_loaded"),
        "latency_s": latency / 1000 if latency is not None else None,
        "mcp_queries": metrics.get("mcp_query_calls"),
        "iterations": metrics.get("iterations"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "total_tokens": metrics.get("total_tokens"),
    }


def _mode_parts(value: str) -> tuple[str, str, str]:
    mode, separator, variant = value.partition("@")
    if not separator:
        return mode, "—", "—"
    parts = variant.split("--")
    if len(parts) >= 2:
        return mode, parts[0], parts[1]
    method = next(
        (name for name in ("no-context", "compact", "full")
         if f"-{name}-" in variant),
        "unknown",
    )
    return mode, method, "unknown"


def _aggregate(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        groups[(point["mode"], point["variant"])].append(point)
    existing = {
        (point["case_id"], point["repetition"]): point["accuracy"]
        for point in points
        if point["mode"] == "existing_skills" and point["accuracy"] is not None
    }
    output = []
    for (mode, variant), group in sorted(groups.items()):
        first = group[0]
        accuracy_values = [point["accuracy"] for point in group if point["accuracy"] is not None]
        format_values = [point["format_valid"] for point in group if point["format_valid"] is not None]
        judge_values = [point["llm_judge_accuracy"] for point in group if point["llm_judge_accuracy"] is not None]
        any_skill_values = [point["any_skill_loaded"] for point in group if point["any_skill_loaded"] is not None]
        expected_values = [point["expected_skill_loaded"] for point in group if point["expected_skill_loaded"] is not None]
        generated_values = [point["generated_skill_loaded"] for point in group if point["generated_skill_loaded"] is not None]
        accuracy = _mean(accuracy_values)
        pairs = [
            (point["accuracy"], existing[(point["case_id"], point["repetition"])])
            for point in group
            if point["accuracy"] is not None
            and (point["case_id"], point["repetition"]) in existing
        ] if mode != "existing_skills" else []
        output.append({
            "mode": mode,
            "source": first["source"],
            "variant": variant,
            "skill": first["skill"],
            "method": first["method"],
            "detail": first["detail"],
            "runs": len(group),
            "accuracy": accuracy,
            "accuracy_correct": sum(accuracy_values),
            "accuracy_total": len(accuracy_values),
            "format_valid": _mean(format_values),
            "format_valid_count": sum(format_values),
            "format_valid_total": len(format_values),
            "llm_judge_accuracy": _mean(judge_values),
            "llm_judge_passed": sum(judge_values),
            "llm_judge_scored": len(judge_values),
            "n_pairs": len(pairs),
            "delta_pp": 100 * _mean(a - b for a, b in pairs) if pairs else None,
            "any_skill_loaded": _mean(any_skill_values),
            "any_skill_loaded_count": sum(any_skill_values),
            "any_skill_loaded_total": len(any_skill_values),
            "expected_skill_loaded": _mean(expected_values),
            "expected_skill_loaded_count": sum(expected_values),
            "expected_skill_loaded_total": len(expected_values),
            "generated_skill_loaded": _mean(generated_values),
            "generated_skill_loaded_count": sum(generated_values),
            "generated_skill_loaded_total": len(generated_values),
            "median_latency_s": _median(point["latency_s"] for point in group),
            "median_mcp_queries": _median(point["mcp_queries"] for point in group),
            "median_iterations": _median(point["iterations"] for point in group),
            "median_input_tokens": _median(point["prompt_tokens"] for point in group),
            "median_output_tokens": _median(point["completion_tokens"] for point in group),
            "median_total_tokens": _median(point["total_tokens"] for point in group),
        })
    return output


def _mean(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _median(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return median(present) if present else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one benchmark HTML report")
    parser.add_argument("result_dir")
    args = parser.parse_args(argv)
    try:
        print(write_report_from_files(args.result_dir))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"benchmark report failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
