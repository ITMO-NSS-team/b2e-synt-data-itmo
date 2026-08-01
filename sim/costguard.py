"""Cost guard: a hard ceiling per experiment, checked before dispatch.

Why "before dispatch" is the load-bearing phrase
------------------------------------------------
A guard that trips *after* the spend has already happened is an accountant, not
a guard. The projection is therefore computed and logged before a batch starts,
the ceiling is compared against the projection, and a batch whose projection
exceeds its ceiling never dispatches a single request.

During a run, the guard is consulted before every model call, so a batch that
drifts past its projection stops mid-flight rather than completing expensively.

Honesty about the numbers
-------------------------
Anthropic does not return a price with a response. Every figure here is a
projection from a configured rate table (``sim.agent.llm.PRICES_USD_PER_MTOK``),
not a measurement, and the API says so in its field names.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from sim.agent.llm import DEFAULT_PRICE, PRICES_USD_PER_MTOK


class CostCeilingExceeded(RuntimeError):
    """Raised before dispatch when a run would breach its ceiling."""


class KillSwitchEngaged(RuntimeError):
    """Raised when an operator has stopped the experiment."""


@dataclass
class Budget:
    """Ceiling for one experiment. Both limits apply; whichever binds first wins."""

    max_tokens: int | None = None
    max_usd: float | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is None and self.max_usd is None:
            raise ValueError(
                "an experiment must declare a ceiling: set max_tokens, max_usd, "
                "or both. An unbounded batch on a metered API is not a default "
                "anyone should get by omission."
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_usd is not None and self.max_usd <= 0:
            raise ValueError("max_usd must be positive")


@dataclass
class Projection:
    """What a batch is expected to cost, computed before anything is spent."""

    questions: int
    expected_calls_per_question: int
    expected_prompt_tokens_per_call: int
    expected_completion_tokens_per_call: int
    model_id: str

    @property
    def total_calls(self) -> int:
        return self.questions * self.expected_calls_per_question

    @property
    def projected_prompt_tokens(self) -> int:
        return self.total_calls * self.expected_prompt_tokens_per_call

    @property
    def projected_completion_tokens(self) -> int:
        return self.total_calls * self.expected_completion_tokens_per_call

    @property
    def projected_tokens(self) -> int:
        return self.projected_prompt_tokens + self.projected_completion_tokens

    @property
    def projected_usd(self) -> float:
        prompt_rate, completion_rate = PRICES_USD_PER_MTOK.get(
            self.model_id, DEFAULT_PRICE)
        return (self.projected_prompt_tokens * prompt_rate
                + self.projected_completion_tokens * completion_rate) / 1_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "expected_calls_per_question": self.expected_calls_per_question,
            "total_calls": self.total_calls,
            "projected_prompt_tokens": self.projected_prompt_tokens,
            "projected_completion_tokens": self.projected_completion_tokens,
            "projected_tokens": self.projected_tokens,
            "projected_usd": round(self.projected_usd, 4),
            "model_id": self.model_id,
            "basis": "configured rate table, not a measured price",
        }


class CostGuard:
    """Tracks spend for one experiment and refuses to exceed its budget."""

    def __init__(self, budget: Budget, *, experiment_id: str) -> None:
        self.budget = budget
        self.experiment_id = experiment_id
        self._lock = threading.Lock()
        self.spent_tokens = 0
        self.spent_usd = 0.0
        self.calls = 0
        self._killed = False
        self._kill_reason = ""

    # ---------------------------------------------------------- pre-dispatch

    def check_projection(self, projection: Projection) -> dict[str, Any]:
        """Compare a projection against the ceiling. Raises if it would breach.

        Returns the projection record that the caller must log before starting.
        """
        record = projection.as_dict()
        breaches = []
        if (self.budget.max_tokens is not None
                and projection.projected_tokens > self.budget.max_tokens):
            breaches.append(
                f"projected {projection.projected_tokens:,} tokens exceeds "
                f"ceiling {self.budget.max_tokens:,}")
        if (self.budget.max_usd is not None
                and projection.projected_usd > self.budget.max_usd):
            breaches.append(
                f"projected ${projection.projected_usd:.2f} exceeds "
                f"ceiling ${self.budget.max_usd:.2f}")
        record["ceiling_tokens"] = self.budget.max_tokens
        record["ceiling_usd"] = self.budget.max_usd
        record["approved"] = not breaches
        if breaches:
            record["breaches"] = breaches
            raise CostCeilingExceeded(
                f"experiment {self.experiment_id} refused before dispatch: "
                + "; ".join(breaches))
        return record

    # ------------------------------------------------------------- in-flight

    def check_before_call(self) -> None:
        """Consulted before every model call."""
        with self._lock:
            if self._killed:
                raise KillSwitchEngaged(
                    f"experiment {self.experiment_id} stopped: {self._kill_reason}")
            if (self.budget.max_tokens is not None
                    and self.spent_tokens >= self.budget.max_tokens):
                raise CostCeilingExceeded(
                    f"experiment {self.experiment_id} hit its token ceiling "
                    f"({self.spent_tokens:,} / {self.budget.max_tokens:,})")
            if (self.budget.max_usd is not None
                    and self.spent_usd >= self.budget.max_usd):
                raise CostCeilingExceeded(
                    f"experiment {self.experiment_id} hit its USD ceiling "
                    f"(${self.spent_usd:.2f} / ${self.budget.max_usd:.2f})")

    def record(self, *, tokens: int, usd: float) -> None:
        with self._lock:
            self.spent_tokens += int(tokens)
            self.spent_usd += float(usd)
            self.calls += 1

    # ----------------------------------------------------------- kill switch

    def kill(self, reason: str = "operator stop") -> None:
        with self._lock:
            self._killed = True
            self._kill_reason = reason

    def resume(self) -> None:
        with self._lock:
            self._killed = False
            self._kill_reason = ""

    @property
    def killed(self) -> bool:
        return self._killed

    def status(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "calls": self.calls,
            "spent_tokens": self.spent_tokens,
            "spent_usd": round(self.spent_usd, 4),
            "ceiling_tokens": self.budget.max_tokens,
            "ceiling_usd": self.budget.max_usd,
            "killed": self._killed,
            "kill_reason": self._kill_reason,
        }


#: Process-wide registry so the admin UI's kill switch can reach a running batch.
_GUARDS: dict[str, CostGuard] = {}
_GUARDS_LOCK = threading.Lock()


def register(guard: CostGuard) -> CostGuard:
    with _GUARDS_LOCK:
        _GUARDS[guard.experiment_id] = guard
    return guard


def get(experiment_id: str) -> CostGuard | None:
    with _GUARDS_LOCK:
        return _GUARDS.get(experiment_id)


def all_guards() -> list[CostGuard]:
    with _GUARDS_LOCK:
        return list(_GUARDS.values())
