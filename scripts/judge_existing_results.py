"""Run only the LLM answer judge over an existing benchmark result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.benchmark.env import load_env
from sim.benchmark.judge import (
    PROMPT_SHA256,
    PROMPT_VERSION,
    AnswerJudge,
    from_env,
)

ELIGIBLE_STATUSES = {"completed", "normalization_pending"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error.msg}") from error
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _targets(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    responses = _jsonl(source / "responses.jsonl")
    scores = _jsonl(source / "scores.jsonl")
    score_ids = {row["run_id"] for row in scores}
    targets = [
        row for row in responses
        if row.get("status") in ELIGIBLE_STATUSES
        and (row.get("case_snapshot") or {}).get("category") == "answerable"
    ]
    missing = [row["run_id"] for row in targets if row["run_id"] not in score_ids]
    if missing:
        raise ValueError(f"eligible responses without scores: {missing[:3]}")
    return targets, scores


def _is_success(record: dict[str, Any]) -> bool:
    verdict = record.get("llm_judge") or {}
    return (
        verdict.get("applicable") is True
        and isinstance(verdict.get("pass"), bool)
        and verdict.get("error") is None
    )


def _evaluate(
    response: dict[str, Any],
    judge: AnswerJudge,
    *,
    attempts: int,
    retry_delay: float,
) -> dict[str, Any]:
    started = time.monotonic()
    snapshot = response["case_snapshot"]
    error: Exception | None = None
    verdict: dict[str, Any] | None = None
    used_attempts = 0
    for used_attempts in range(1, attempts + 1):
        try:
            verdict = judge.evaluate(
                query=snapshot["original_query"],
                reference=snapshot["gold_answer"],
                comparison=snapshot["gold_comparison"],
                candidate=response["response"]["raw_answer"],
            )
            break
        except Exception as caught:  # Each failure is persisted and resumable.
            error = caught
            if used_attempts < attempts:
                time.sleep(retry_delay * used_attempts)
    if verdict is None:
        verdict = {
            "applicable": True,
            "pass": None,
            "scores": None,
            "explanation": None,
            "model": getattr(judge, "model", None),
            "prompt_version": PROMPT_VERSION,
            "usage": None,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "run_id": response["run_id"],
        "case_id": response["case_id"],
        "mode": response["mode"],
        "repetition": response["repetition"],
        "status": response["status"],
        "evaluated_at": _now(),
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "attempts": used_attempts,
        "llm_judge": verdict,
    }


def _summary(
    targets: list[dict[str, Any]], latest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    for response in targets:
        mode = response["mode"]
        group = by_mode.setdefault(mode, {"runs": 0, "scored": 0, "passed": 0, "errors": 0})
        group["runs"] += 1
        record = latest.get(response["run_id"])
        if not record or not _is_success(record):
            group["errors"] += 1
            continue
        group["scored"] += 1
        group["passed"] += int(record["llm_judge"]["pass"])
    for group in by_mode.values():
        group["accuracy"] = (
            group["passed"] / group["scored"] if group["scored"] else None
        )
    scored = sum(group["scored"] for group in by_mode.values())
    passed = sum(group["passed"] for group in by_mode.values())
    errors = sum(group["errors"] for group in by_mode.values())
    return {
        "runs": len(targets),
        "scored": scored,
        "passed": passed,
        "errors": errors,
        "accuracy": passed / scored if scored else None,
        "by_mode": dict(sorted(by_mode.items())),
    }


def run_posthoc(
    source: Path,
    output: Path,
    judge: AnswerJudge,
    *,
    workers: int = 2,
    attempts: int = 1,
    retry_delay: float = 15.0,
    limit: int | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets, scores = _targets(source)
    if limit is not None:
        targets = targets[:limit]

    identity = {
        "source": str(source),
        "source_responses_sha256": _sha256(source / "responses.jsonl"),
        "source_scores_sha256": _sha256(source / "scores.jsonl"),
        "model": getattr(judge, "model", None),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("existing post-hoc manifest has different source or judge")
    else:
        manifest = {
            "status": "running",
            "started_at": _now(),
            "identity": identity,
            "workers": workers,
            "attempts": attempts,
            "runs": len(targets),
        }
        _write_json(manifest_path, manifest)

    judgments_path = output / "judgments.jsonl"
    previous = _jsonl(judgments_path)
    latest = {row["run_id"]: row for row in previous}
    pending = [row for row in targets if not _is_success(latest.get(row["run_id"], {}))]
    print(f"eligible={len(targets)} completed={len(targets) - len(pending)} pending={len(pending)}")

    completed = len(targets) - len(pending)
    with judgments_path.open("a", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _evaluate, response, judge,
                    attempts=attempts, retry_delay=retry_delay,
                ): response["run_id"]
                for response in pending
            }
            for future in as_completed(futures):
                record = future.result()
                checkpoint.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                checkpoint.flush()
                latest[record["run_id"]] = record
                completed += 1
                if completed % 10 == 0 or completed == len(targets):
                    print(f"progress={completed}/{len(targets)}")

    summary = _summary(targets, latest)
    _write_json(output / "summary.json", summary)

    target_ids = {row["run_id"] for row in targets}
    augmented = []
    for score in scores:
        copy = json.loads(json.dumps(score))
        if score["run_id"] in target_ids and score["run_id"] in latest:
            verdict = latest[score["run_id"]]["llm_judge"]
            copy["llm_judge"] = verdict
            copy.setdefault("metrics", {})["llm_judge_accuracy"] = (
                int(verdict["pass"]) if isinstance(verdict.get("pass"), bool) else None
            )
        augmented.append(copy)
    _write_jsonl(output / "scores-with-llm-judge.jsonl", augmented)

    manifest.update({
        "status": "complete" if summary["errors"] == 0 else "partial",
        "completed_at": _now(),
    })
    _write_json(manifest_path, manifest)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Judge saved benchmark responses without rerunning the B2E agent.",
    )
    result.add_argument("result_dir", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--attempts", type=int, default=1)
    result.add_argument("--retry-delay", type=float, default=15.0)
    result.add_argument("--limit", type=int)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    targets, _scores = _targets(args.result_dir)
    if args.limit is not None:
        targets = targets[: args.limit]
    if args.dry_run:
        print(json.dumps({"eligible": len(targets)}, indent=2))
        return 0
    if args.workers < 1 or args.attempts < 1 or args.retry_delay < 0:
        raise ValueError("workers and attempts must be positive; retry-delay must be non-negative")

    load_env(ROOT / "deploy/.env")
    load_env(ROOT / "sim/benchmark/.env")
    judge = from_env()
    if judge is None:
        raise ValueError("BENCH_LLM_JUDGE_ENABLED must be true")
    output = args.output_dir or args.result_dir / "posthoc-llm-judge"
    summary = run_posthoc(
        args.result_dir, output, judge,
        workers=args.workers,
        attempts=args.attempts,
        retry_delay=args.retry_delay,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
