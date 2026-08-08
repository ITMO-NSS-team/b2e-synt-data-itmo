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

#: Cache pricing, as multipliers on the model's base input rate. A read is a
#: tenth of a fresh token; a write costs *more* than one, which is easy to get
#: backwards because both have "cache" in the name. The write multiplier is the
#: one-hour figure — that is the TTL this stack's sessions actually report
#: (``ephemeral_1h_input_tokens`` in the CLI's usage envelope).
#:
#: Validated rather than assumed: with these three rates the projection below
#: reproduces the measured batch (92 calls, 7 turns, 2026-08-08) at $0.6672
#: against the $0.6717 the CLI itself reported — 0.7 % apart. Pricing every
#: prompt token at the base rate gives $2.52 for the same batch.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 2.00

#: Per model call, measured from the LLM spans of 92 calls across 7 turns on
#: 2026-08-08 — questions ranging from a two-line catalogue lookup to an
#: open-ended analytics request, 5 to 26 calls per turn.
#:
#: These are means, and the spread is wide: the prompt runs 12 100 to 51 479
#: tokens per call (median 23 412, p90 40 630), because the cached prefix grows
#: as a turn proceeds. A projection built from them is a central estimate of a
#: batch total, not a worst case for any single call.
#:
#: The previous value of this first constant was 6 000, and nothing could check
#: it: per-call counts did not exist, and the turn-level figure excluded the
#: cached prefix — which is 94 % of the prompt here.
MEASURED_PROMPT_TOKENS_PER_CALL = 25_791
MEASURED_CACHE_READ_TOKENS_PER_CALL = 24_178
MEASURED_CACHE_WRITE_TOKENS_PER_CALL = 1_605
MEASURED_COMPLETION_TOKENS_PER_CALL = 323


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
    #: Components *of* the prompt above, not additions to it. Defaulted to the
    #: measured split so a caller written before the split existed does not
    #: silently fall back to pricing every token as fresh — which was the old
    #: behaviour and wrong by a factor of four.
    expected_cache_read_tokens_per_call: int = MEASURED_CACHE_READ_TOKENS_PER_CALL
    expected_cache_write_tokens_per_call: int = MEASURED_CACHE_WRITE_TOKENS_PER_CALL

    @property
    def total_calls(self) -> int:
        return self.questions * self.expected_calls_per_question

    @property
    def projected_prompt_tokens(self) -> int:
        return self.total_calls * self.expected_prompt_tokens_per_call

    @property
    def projected_cache_read_tokens(self) -> int:
        return self.total_calls * self.expected_cache_read_tokens_per_call

    @property
    def projected_cache_write_tokens(self) -> int:
        return self.total_calls * self.expected_cache_write_tokens_per_call

    @property
    def projected_fresh_prompt_tokens(self) -> int:
        """Prompt tokens billed at the full input rate — what is left after the
        cached prefix. On this stack it is almost nothing: 8 tokens per call
        against 24 178 read from cache."""
        return max(0, self.projected_prompt_tokens
                   - self.projected_cache_read_tokens
                   - self.projected_cache_write_tokens)

    @property
    def projected_completion_tokens(self) -> int:
        return self.total_calls * self.expected_completion_tokens_per_call

    @property
    def projected_tokens(self) -> int:
        """Everything the model will read and write, cached prefix included.

        Deliberately the same accounting `record()` uses, because both sides of
        the token ceiling have to mean the same thing. Cost is where the cache
        distinction belongs; volume is volume.
        """
        return self.projected_prompt_tokens + self.projected_completion_tokens

    @property
    def projected_usd(self) -> float:
        input_rate, completion_rate = PRICES_USD_PER_MTOK.get(
            self.model_id, DEFAULT_PRICE)
        return (self.projected_fresh_prompt_tokens * input_rate
                + self.projected_cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER
                + self.projected_cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
                + self.projected_completion_tokens * completion_rate) / 1_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "expected_calls_per_question": self.expected_calls_per_question,
            "total_calls": self.total_calls,
            "projected_prompt_tokens": self.projected_prompt_tokens,
            "projected_cache_read_tokens": self.projected_cache_read_tokens,
            "projected_cache_write_tokens": self.projected_cache_write_tokens,
            "projected_fresh_prompt_tokens": self.projected_fresh_prompt_tokens,
            "projected_completion_tokens": self.projected_completion_tokens,
            "projected_tokens": self.projected_tokens,
            "projected_usd": round(self.projected_usd, 4),
            "model_id": self.model_id,
            "basis": "configured rate table with cache multipliers, "
                     "calibrated 2026-08-08 against 92 measured calls; "
                     "not a measured price",
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
