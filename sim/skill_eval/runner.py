from __future__ import annotations

from collections import defaultdict
from typing import Any

from sim.skill_eval.catalog import CatalogStrategy
from sim.skill_eval.loggers.base import EvalLogger
from sim.skill_eval.scoring.registry import ScorerRegistry
from sim.skill_eval.stand import StandClient
from sim.skill_eval.types import EvalCase, RunSummary, TurnResult


class EvalRunner:
    def __init__(
        self,
        catalog: CatalogStrategy,
        scorers: ScorerRegistry,
        loggers: EvalLogger,
        stand: StandClient,
        *,
        eval_id: str,
        hydra_run: str,
        ignore_snapshot: bool = False,
    ) -> None:
        self.catalog = catalog
        self.scorers = scorers
        self.loggers = loggers
        self.stand = stand
        self.eval_id = eval_id
        self.hydra_run = hydra_run
        self.ignore_snapshot = ignore_snapshot

    def run(self, cases: list[EvalCase]) -> RunSummary:
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                self.catalog.prepare(case)
                spec = self.catalog.session_spec(
                    case, eval_id=self.eval_id, hydra_run=self.hydra_run)
                turn = self.stand.run(case, spec)
                snapshot_ok = self._snapshot_ok(case, turn)
                score = self.scorers.score(case, turn, snapshot_ok=snapshot_ok)
                record = _record(
                    case, turn, score,
                    eval_id=self.eval_id,
                    catalog=self.catalog.name,
                    snapshot_ok=snapshot_ok,
                )
                records.append(record)
                self.loggers.log_case(case, turn, score, record=record)
        finally:
            self.catalog.teardown()
        summary = _summarise(
            records, eval_id=self.eval_id, catalog_name=self.catalog.name)
        self.loggers.log_run(summary)
        return summary

    def _snapshot_ok(self, case: EvalCase, turn: TurnResult) -> bool:
        if self.ignore_snapshot:
            return True
        live = turn.live_snapshot_id
        expected = case.snapshot_id
        if not expected or not live:
            return False
        return live == expected


def _record(
    case: EvalCase, turn: TurnResult, score, *,
    eval_id: str, catalog: str, snapshot_ok: bool,
) -> dict[str, Any]:
    return {
        "eval_id": eval_id,
        "catalog": catalog,
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "actor": case.runtime_actor_employee_id,
        "expected_skill": case.expected_skill,
        "gold_snapshot_id": case.snapshot_id,
        "live_snapshot_id": turn.live_snapshot_id,
        "snapshot_ok": snapshot_ok,
        "answer": turn.answer,
        "session_id": turn.session_id,
        "trace_id": turn.trace_id,
        "retrieved_skills": turn.retrieved_skills,
        "expected_skill_retrieved": score.routing_hit,
        "task_success": score.task_success,
        "skipped_numeric": score.skipped_numeric,
        "reasons": score.reasons,
        "heimdall_calls": turn.heimdall_calls,
        "tool_calls": turn.tool_calls,
        "total_tokens": turn.total_tokens,
        "latency_ms": turn.latency_ms,
        "error": turn.error,
    }


def _summarise(
    records: list[dict[str, Any]], *, eval_id: str, catalog_name: str,
) -> RunSummary:
    n = len(records) or 1
    successes = [row for row in records if row["task_success"]]
    routed = [row for row in records if row["expected_skill_retrieved"] is not None]
    hits = [row for row in routed if row["expected_skill_retrieved"]]
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_cat[row["category"]].append(row)
    per_category = {
        category: {
            "task_success_rate": _rate(group, "task_success"),
            "n": float(len(group)),
        }
        for category, group in by_cat.items()
    }
    return RunSummary(
        eval_id=eval_id,
        catalog_name=catalog_name,
        n_cases=len(records),
        task_success_rate=len(successes) / n,
        routing_accuracy=(len(hits) / len(routed)) if routed else None,
        mean_heimdall_calls=_mean(records, "heimdall_calls"),
        mean_tokens=_mean(records, "total_tokens"),
        mean_latency_ms=_mean(records, "latency_ms"),
        per_category=per_category,
        records=records,
    )


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row.get(key)) / len(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0) for row in rows]
    return sum(values) / len(values) if values else 0.0
