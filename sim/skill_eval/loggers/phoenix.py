from __future__ import annotations

from typing import Any

import httpx

from sim.research.phoenix_client import PhoenixClient, PhoenixUnavailable
from sim.skill_eval.loggers.base import EvalLogger
from sim.skill_eval.types import CaseScore, EvalCase, RunSummary, TurnResult


class PhoenixLogger(EvalLogger):
    def __init__(
        self,
        project: str = "b2e-itmo",
        client: PhoenixClient | None = None,
    ) -> None:
        self._project = project
        self._phoenix = client

    def bind_http(self, http: httpx.Client) -> None:
        self._phoenix = PhoenixClient(
            str(http.base_url), project=self._project, client=http)

    def log_case(
        self, case: EvalCase, turn: TurnResult, score: CaseScore,
        *, record: dict[str, Any],
    ) -> None:
        del record
        if self._phoenix is None:
            return
        span_id = turn.root_span_id
        if not span_id:
            return
        label = "pass" if score.task_success else "fail"
        explanation = (
            f"{case.category} routing={score.routing_hit} "
            f"reasons={score.reasons}"
        )
        try:
            self._phoenix.annotate_span(
                span_id=span_id,
                name="skill_eval",
                label=label,
                score=1.0 if score.task_success else 0.0,
                explanation=explanation,
                annotator="skill_eval",
            )
        except PhoenixUnavailable:
            return

    def log_run(self, summary: RunSummary) -> None:
        del summary
