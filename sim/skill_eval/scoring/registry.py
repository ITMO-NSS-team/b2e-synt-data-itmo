from __future__ import annotations

from sim.skill_eval.scoring.base import CategoryScorer
from sim.skill_eval.types import CaseScore, EvalCase, TurnResult


class ScorerRegistry:
    def __init__(self, scorers: dict[str, CategoryScorer]) -> None:
        self._scorers = dict(scorers)

    def score(
        self, case: EvalCase, turn: TurnResult, *, snapshot_ok: bool,
    ) -> CaseScore:
        scorer = self._scorers.get(case.category)
        if scorer is None:
            known = ", ".join(sorted(self._scorers)) or "(none)"
            raise KeyError(
                f"no CategoryScorer for {case.category!r}; registered: {known}")
        return scorer.score(case, turn, snapshot_ok=snapshot_ok)
