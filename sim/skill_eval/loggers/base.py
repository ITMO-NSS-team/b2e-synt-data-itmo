from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sim.skill_eval.types import CaseScore, EvalCase, RunSummary, TurnResult


class EvalLogger(ABC):
    @abstractmethod
    def log_case(
        self, case: EvalCase, turn: TurnResult, score: CaseScore,
        *, record: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    def log_run(self, summary: RunSummary) -> None:
        ...


class CompositeLogger(EvalLogger):
    def __init__(self, loggers: list[EvalLogger]) -> None:
        self.loggers = list(loggers)

    def log_case(
        self, case: EvalCase, turn: TurnResult, score: CaseScore,
        *, record: dict[str, Any],
    ) -> None:
        for logger in self.loggers:
            logger.log_case(case, turn, score, record=record)

    def log_run(self, summary: RunSummary) -> None:
        for logger in self.loggers:
            logger.log_run(summary)
