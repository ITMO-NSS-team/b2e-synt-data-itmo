"""Self-contained Plotly report for one benchmark evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import median
from string import Template
from typing import Any, Iterable

from plotly.offline import get_plotlyjs


_ASSETS = Path(__file__).with_name("report_assets")


def write_report(result_dir: str | Path, results: Iterable[Any]) -> Path:
    """Write a self-contained report beside the benchmark JSON artifacts."""
    root = Path(result_dir)
    rows = [_result_row(result) for result in results]
    return _publish(_write(root, rows), root.name)


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
    return _publish(_write(root, rows), root.name)


def _publish(path: Path, eval_id: str) -> Path:
    if public_root := (os.getenv("BENCHMARK_REPORT_DIR") or "").strip():
        public_path = Path(public_root) / eval_id / "report.html"
        public_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, public_path)
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
    points = [_point(row) for row in rows]
    payload = json.dumps(
        {
            "eval_id": root.name,
            "runs": points,
            "aggregates": _aggregate(points),
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


def _point(row: dict[str, Any]) -> dict[str, Any]:
    mode, method, detail = _mode_parts(str(row["mode"]))
    metrics = row["metrics"]
    latency = metrics.get("latency_ms")
    return {
        "run_id": row["run_id"],
        "case_id": row["case_id"],
        "mode": mode,
        "method": method,
        "detail": detail,
        "repetition": row["repetition"],
        "status": row["status"],
        "accuracy": metrics.get("answer_accuracy"),
        "any_skill_loaded": metrics.get("any_skill_loaded"),
        "expected_skill_loaded": metrics.get("expected_skill_loaded"),
        "latency_s": latency / 1000 if latency is not None else None,
        "mcp_queries": metrics.get("mcp_query_calls"),
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
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        groups[(point["mode"], point["method"], point["detail"])].append(point)
    existing = [
        point["accuracy"] for point in points
        if point["mode"] == "existing_skills" and point["accuracy"] is not None
    ]
    baseline = sum(existing) / len(existing) if existing else None
    output = []
    detail_order = {"—": 0, "abstract": 1, "medium": 2, "detailed": 3}
    for (mode, method, detail), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0], item[0][1], detail_order.get(item[0][2], 99),
        ),
    ):
        accuracy = _mean(point["accuracy"] for point in group)
        output.append({
            "mode": mode,
            "method": method,
            "detail": detail,
            "runs": len(group),
            "accuracy": accuracy,
            "delta_pp": (
                (accuracy - baseline) * 100
                if accuracy is not None and baseline is not None
                and mode != "existing_skills" else None
            ),
            "any_skill_loaded": _mean(
                point["any_skill_loaded"] for point in group
            ),
            "expected_skill_loaded": _mean(
                point["expected_skill_loaded"] for point in group
            ),
            "median_latency_s": _median(point["latency_s"] for point in group),
            "median_mcp_queries": _median(point["mcp_queries"] for point in group),
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
