from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sim.skill_eval.loggers.base import EvalLogger
from sim.skill_eval.types import CaseScore, EvalCase, RunSummary, TurnResult


class JsonlLogger(EvalLogger):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._summary_path = self.path.with_suffix(".summary.json")

    def log_case(
        self, case: EvalCase, turn: TurnResult, score: CaseScore,
        *, record: dict[str, Any],
    ) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_run(self, summary: RunSummary) -> None:
        payload = {
            "eval_id": summary.eval_id,
            "catalog": summary.catalog_name,
            "n_cases": summary.n_cases,
            "task_success_rate": summary.task_success_rate,
            "routing_accuracy": summary.routing_accuracy,
            "mean_heimdall_calls": summary.mean_heimdall_calls,
            "mean_tokens": summary.mean_tokens,
            "mean_latency_ms": summary.mean_latency_ms,
            "per_category": summary.per_category,
        }
        self._summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
