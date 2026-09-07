from __future__ import annotations

from abc import ABC, abstractmethod

from sim.skill_eval.types import CaseScore, EvalCase, TurnResult


class CategoryScorer(ABC):
    @abstractmethod
    def score(self, case: EvalCase, turn: TurnResult, *, snapshot_ok: bool) -> CaseScore:
        ...
