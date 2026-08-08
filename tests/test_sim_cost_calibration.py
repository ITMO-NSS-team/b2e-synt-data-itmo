"""What a batch is projected to cost, checked against what one actually cost.

Until 2026-08-08 the projection assumed 6 000 prompt tokens per model call. That
number could not be checked, because per-call token counts were not recorded
anywhere — and the turn-level figure that was recorded excluded the cached
prefix, which is 94 % of the prompt on this stack.

With LLM spans in place the numbers here are measured rather than assumed: 92
model calls over 7 turns on 2026-08-08, spanning questions from a two-line
catalogue lookup to an open-ended analytics request (5 to 26 calls per turn).
The constants under test are that measurement, and this file is what stops them
drifting away from it silently.
"""
from __future__ import annotations

import pytest

from sim.costguard import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    MEASURED_CACHE_READ_TOKENS_PER_CALL,
    MEASURED_CACHE_WRITE_TOKENS_PER_CALL,
    MEASURED_COMPLETION_TOKENS_PER_CALL,
    MEASURED_PROMPT_TOKENS_PER_CALL,
    Projection,
)

MODEL = "claude-haiku-4-5-20251001"

#: The batch the constants were taken from: 92 calls, and what it really cost
#: according to the CLI's own closing figures, summed over the 7 turns.
MEASURED_CALLS = 92
MEASURED_USD = 0.6717


def _measured_projection(calls: int = MEASURED_CALLS) -> Projection:
    return Projection(
        questions=calls, expected_calls_per_question=1,
        expected_prompt_tokens_per_call=MEASURED_PROMPT_TOKENS_PER_CALL,
        expected_completion_tokens_per_call=MEASURED_COMPLETION_TOKENS_PER_CALL,
        model_id=MODEL)


def test_the_projection_reproduces_a_batch_that_actually_ran():
    """The whole point of a pre-dispatch estimate. Within 10 % of the real
    invoice for the batch its constants came from; measured error was 0.7 %."""
    projected = _measured_projection().projected_usd
    assert projected == pytest.approx(MEASURED_USD, rel=0.10)


def test_pricing_every_prompt_token_as_fresh_would_be_wildly_wrong():
    """Why the cache split has to exist at all: 94 % of the prompt is a cache
    read, billed at a tenth of the input rate. Charging all of it at full price
    over-projects by nearly 4x and refuses batches that are affordable."""
    flat = (MEASURED_CALLS
            * (MEASURED_PROMPT_TOKENS_PER_CALL * 1.00
               + MEASURED_COMPLETION_TOKENS_PER_CALL * 5.00) / 1_000_000)
    assert flat > MEASURED_USD * 3


def test_a_cache_read_is_not_billed_as_a_fresh_prompt_token():
    cached = Projection(
        questions=1, expected_calls_per_question=1,
        expected_prompt_tokens_per_call=10_000,
        expected_completion_tokens_per_call=0,
        expected_cache_read_tokens_per_call=10_000,
        expected_cache_write_tokens_per_call=0,
        model_id=MODEL)
    fresh = Projection(
        questions=1, expected_calls_per_question=1,
        expected_prompt_tokens_per_call=10_000,
        expected_completion_tokens_per_call=0,
        expected_cache_read_tokens_per_call=0,
        expected_cache_write_tokens_per_call=0,
        model_id=MODEL)
    assert cached.projected_usd == pytest.approx(fresh.projected_usd
                                                 * CACHE_READ_MULTIPLIER)


def test_writing_the_cache_costs_more_than_reading_a_fresh_token():
    """A 1-hour cache write is billed above the base input rate, not below it.
    Treating it as cheap because it has "cache" in the name would understate the
    one part of the prompt that is genuinely expensive."""
    assert CACHE_WRITE_MULTIPLIER > 1.0 > CACHE_READ_MULTIPLIER


def test_projected_tokens_still_counts_everything_the_guard_will_record():
    """`projected_tokens` is compared against a token ceiling that `record()`
    fills with real totals — cached prefix included. If the projection counted
    only fresh tokens the two sides of the same ceiling would mean different
    things."""
    projection = _measured_projection(calls=10)
    assert projection.projected_prompt_tokens == 10 * MEASURED_PROMPT_TOKENS_PER_CALL


def test_the_shipped_calibration_is_the_one_that_was_measured():
    """Pins the constants to the observation. A future edit that nudges them
    without a fresh measurement has to delete this test to do it, which is the
    point."""
    assert MEASURED_PROMPT_TOKENS_PER_CALL == 25_791
    assert MEASURED_CACHE_READ_TOKENS_PER_CALL == 24_178
    assert MEASURED_CACHE_WRITE_TOKENS_PER_CALL == 1_605
    assert MEASURED_COMPLETION_TOKENS_PER_CALL == 323


def test_the_cache_split_cannot_exceed_the_prompt_it_is_part_of():
    """Cache read plus cache write are components of the prompt, not additions
    to it. A calibration where they exceed it would bill tokens twice."""
    assert (MEASURED_CACHE_READ_TOKENS_PER_CALL
            + MEASURED_CACHE_WRITE_TOKENS_PER_CALL) <= MEASURED_PROMPT_TOKENS_PER_CALL


def test_a_projection_without_a_declared_split_assumes_the_measured_one():
    """Callers that predate the split must not silently get the old flat
    pricing, which was the wrong answer by a factor of four."""
    default = Projection(
        questions=1, expected_calls_per_question=1,
        expected_prompt_tokens_per_call=MEASURED_PROMPT_TOKENS_PER_CALL,
        expected_completion_tokens_per_call=MEASURED_COMPLETION_TOKENS_PER_CALL,
        model_id=MODEL)
    assert default.expected_cache_read_tokens_per_call == \
        MEASURED_CACHE_READ_TOKENS_PER_CALL


def test_the_record_says_the_projection_is_cache_aware():
    """The logged projection is what a researcher reads back months later; it
    has to say which pricing model produced the number."""
    record = _measured_projection().as_dict()
    assert record["projected_cache_read_tokens"] == \
        MEASURED_CALLS * MEASURED_CACHE_READ_TOKENS_PER_CALL
    assert "cache" in record["basis"]
