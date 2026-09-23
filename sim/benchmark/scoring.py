"""Deterministic normalization and first-version benchmark metrics."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from .cases import BenchmarkCase
from .contracts import response_schema
from .modes import ModeConfig

_FENCED_JSON = re.compile(r"[\x60]{3}(?:json)?\s*\n(.*?)\n[\x60]{3}", re.DOTALL | re.IGNORECASE)
_REFUSAL_OUTCOMES = frozenset({"access_control", "missing_skill", "out_of_scope"})


@dataclass(frozen=True, slots=True)
class NormalizedAnswer:
    """Result of deterministic JSON extraction from a model answer.

    Attributes:
        value: Object matching the public response schema, or ``None``.
        error: Why extraction or schema validation failed, or ``None``.
    """
    value: dict[str, Any] | None
    error: str | None


def normalize_answer(answer: str, response_contract: dict[str, Any]) -> NormalizedAnswer:
    """Extract a JSON object and validate it against the case contract.

    This deterministic normalizer does not call an LLM. It accepts a direct
    JSON answer, a fenced JSON object, or the first decodable object embedded
    in prose. An unstructured answer is retained raw but receives no fabricated
    normalized value.

    Args:
        answer: Raw final text from the agent turn.
        response_contract: Public contract used to expand the response schema.

    Returns:
        Parsed object when a candidate matches the schema; otherwise
        ``value=None`` and a human-readable ``error``.
    """
    validator = Draft202012Validator(response_schema(response_contract))
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in _json_candidates(answer):
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(candidate)
        errors = list(validator.iter_errors(candidate))
        if not errors:
            return NormalizedAnswer(candidate, None)
    if not parsed:
        return NormalizedAnswer(None, "final answer contains no JSON object")
    first_error = next(validator.iter_errors(parsed[0]), None)
    detail = first_error.message if first_error is not None else "no candidate matched"
    return NormalizedAnswer(None, f"normalized answer violates response_contract: {detail}")


def calculate_metrics(
    case: BenchmarkCase,
    mode: ModeConfig,
    normalized: NormalizedAnswer,
    observations: dict[str, Any],
) -> dict[str, Any]:
    """Score one completed run. Aggregate rates are means of these 0/1 values.

    Args:
        case: Ready authorial case with gold and comparison rules.
        mode: Arm that produced the turn.
        normalized: Extracted JSON, or an extraction failure.
        observations: Trace-derived call counts, errors and timings.

    Returns:
        Metric map. Accuracy fields are ``0``, ``1`` or ``None`` when the
        answer could not be normalized or the metric does not apply.
    """
    actual = normalized.value
    evaluation = case.raw["evaluation_contract"]
    expected_outcome = evaluation["expected_outcome"]
    observed_outcome = (
        _observed_outcome(case.raw["category"], actual, observations)
        if actual is not None else None
    )
    outcome_accuracy = (
        int(observed_outcome == expected_outcome) if actual is not None else None
    )
    exact_match: int | None = None
    if actual is not None and expected_outcome == "answer":
        exact_match = int(
            observed_outcome == "answer" and _results_equal(
                actual.get("result"),
                evaluation["gold_result"],
                evaluation["comparison"],
            )
        )
    if expected_outcome == "answer":
        answer_accuracy = exact_match
    else:
        answer_accuracy = outcome_accuracy
    correct_refusal = (
        outcome_accuracy if expected_outcome in _REFUSAL_OUTCOMES else None
    )
    generated_loaded = mode.strategy.skill_loaded_metric(
        mode,
        case.raw["expected_skills"],
        observations["loaded_skills"],
    )
    return {
        "answer_accuracy": answer_accuracy,
        "exact_match": exact_match,
        "outcome_accuracy": outcome_accuracy,
        "correct_refusal": correct_refusal,
        "generated_skill_loaded": generated_loaded,
        "heimdall_calls": observations["heimdall_calls"],
        "mcp_query_calls": observations["mcp_query_calls"],
        "failed_tool_calls": observations["failed_tool_calls"],
        "total_tokens": observations["total_tokens"],
        "latency_ms": observations["latency_ms"],
        "agent_duration_ms": observations["agent_duration_ms"],
        "tool_time_ms": observations["tool_time_ms"],
    }


def _observed_outcome(
    category: str,
    actual: dict[str, Any],
    observations: dict[str, Any],
) -> str | None:
    """Infer the outcome from result and trace without asking the agent to label it.

    Args:
        category: Authorial case category.
        actual: Normalized ``{result, message}`` object.
        observations: Trace facts used for refusal categories.

    Returns:
        Outcome label, or ``None`` when the traces do not support a label.
    """
    if actual.get("result") is not None:
        return "answer"
    message = actual.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    statuses = set(observations.get("http_statuses") or ())
    errors = {str(item).lower() for item in observations.get("error_codes") or ()}
    rows = observations.get("mcp_query_rows") or []
    if category == "access_control" and (403 in statuses or "forbidden" in errors):
        return "access_control"
    if category == "no_data" and rows and all(value == 0 for value in rows):
        return "no_data"
    if category == "out_of_scope" and observations.get("heimdall_calls", 0) == 0:
        return "out_of_scope"
    if category == "missing_skill" and observations.get("heimdall_calls", 0) > 0:
        return "missing_skill"
    return None


def _json_candidates(text: str):
    decoder = json.JSONDecoder()
    stripped = text.strip()
    attempts = [stripped, *(_FENCED_JSON.findall(text))]
    for attempt in attempts:
        try:
            value = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            yield value
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _results_equal(
    actual: Any,
    expected: Any,
    comparison: dict[str, Any],
) -> bool:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return _value_equal(
            actual, expected, float(comparison["numeric_absolute_tolerance"])
        )
    actual_rows = actual
    expected_rows = expected
    ordered = comparison["ordered"]
    row_key = comparison["row_key"]
    allow_extra = comparison["allow_extra_rows"]
    tolerance = float(comparison["numeric_absolute_tolerance"])
    if not ordered and row_key:
        if any(not isinstance(row, dict) for row in actual_rows + expected_rows):
            return False
        actual_by_key = {_row_key(row, row_key): row for row in actual_rows}
        expected_by_key = {_row_key(row, row_key): row for row in expected_rows}
        if len(actual_by_key) != len(actual_rows) or len(expected_by_key) != len(expected_rows):
            return False
        if allow_extra:
            if not set(expected_by_key) <= set(actual_by_key):
                return False
        elif set(actual_by_key) != set(expected_by_key):
            return False
        return all(
            _value_equal(actual_by_key[key], expected_by_key[key], tolerance)
            for key in expected_by_key
        )
    if allow_extra:
        unmatched = list(actual_rows)
        for expected_row in expected_rows:
            found = next(
                (i for i, row in enumerate(unmatched) if _value_equal(row, expected_row, tolerance)),
                None,
            )
            if found is None:
                return False
            unmatched.pop(found)
        return True
    if len(actual_rows) != len(expected_rows):
        return False
    if not ordered:
        actual_rows = sorted(actual_rows, key=_canonical)
        expected_rows = sorted(expected_rows, key=_canonical)
    return all(
        _value_equal(a, e, tolerance) for a, e in zip(actual_rows, expected_rows)
    )


def _row_key(row: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(_canonical(row.get(key)) for key in keys)


def _value_equal(actual: Any, expected: Any, tolerance: float) -> bool:
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _value_equal(actual[key], expected[key], tolerance) for key in expected
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _value_equal(a, e, tolerance) for a, e in zip(actual, expected)
        )
    return actual == expected


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
