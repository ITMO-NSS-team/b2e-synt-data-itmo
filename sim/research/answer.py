"""The agent's structured tail, and how it is read.

Scoring free prose against a reference is a scorer whose disagreements are
partly its own — ``sim/oracle/labels.py`` makes the same argument about
returning dataclasses rather than sentences. So the agent appends a fenced
block, and correctness becomes a comparison rather than an interpretation.

The block is additive: the prose answer stays, because the presentation rubric
scores it and because a user-facing agent that emitted only YAML would not be
the agent under study.

Tolerant, except about pretending to have read something it did not
------------------------------------------------------------------
The parser forgives shape drift the contract does not show — a YAML block list
for ``ids`` instead of ``[a, b]``, a trailing ``%`` on ``value`` — because a
model that chose a different-but-unambiguous spelling answered correctly and a
scorer that fails it would be measuring the parser, not the agent.

What it will not do is let "present but unreadable" collapse into the same
``None``/``False`` that "absent" or "explicitly null" produce. Those are
different situations: an absent or null field means the model did not attempt
an answer, which is a legitimate outcome the rubric scores as unscored; an
unreadable field (``value: 1/3``, ``refused: maybe``) means it attempted one in
a shape nobody can read, which is a parser failure that must not silently
present as a wrong numeric answer or a `False` refusal. ``ParsedAnswer.value``
and ``.refused`` keep their defaults in both cases — a consumer that only reads
those two fields cannot tell them apart, by design, because for scoring
purposes both are "no reliable answer here". The distinction lives in
``field_errors``: a field name appears there if and only if it was present in
the block and could not be read, which is what lets a caller who cares recover
the three states — absent/null, present-and-read, present-and-unreadable — by
checking ``field_errors`` first and the value second.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sim.agent.prompt import ANSWER_CONTRACT  # noqa: F401  re-exported for tests

_BLOCK = re.compile(r"```answer\s*\n(.*?)```", re.S | re.I)
_NULLS = {"null", "none", "nil", "-", ""}
_FIELD_NAMES = ("verdict", "ids", "value", "refused", "reason")
_TRUE_WORDS = {"true", "да", "yes", "1"}
_FALSE_WORDS = {"false", "нет", "no", "0"}


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    present: bool
    verdict: str | None = None
    ids: list[str] = field(default_factory=list)
    value: float | None = None
    refused: bool = False
    reason: str | None = None
    error: str | None = None
    #: field name -> reason, present only for a field that appeared in the
    #: block but could not be read. Absent or explicitly-null fields never
    #: appear here — see the module docstring for why that is deliberate.
    field_errors: dict[str, str] = field(default_factory=dict)


def _scalar(raw: str) -> str | None:
    text = raw.strip().strip('"').strip("'").strip()
    return None if text.lower() in _NULLS else text


def _number(raw: str) -> tuple[float | None, bool]:
    """Parse a numeric field. Returns ``(value, ok)``.

    ``ok`` is False exactly when the field held text that is neither a
    recognised null spelling nor a number this parser knows how to read — the
    "present but unreadable" case a caller must not confuse with an explicit
    null, which also yields ``value=None`` but with ``ok=True``.
    """
    text = _scalar(raw)
    if text is None:
        return None, True
    as_share = text.endswith("%")
    if as_share:
        text = text[:-1].strip()
    # A Russian-language model writes 0,3125 as often as 0.3125, and thousands
    # separators arrive as spaces or non-breaking spaces.
    cleaned = (text.replace("\xa0", "").replace(" ", "")
                   .replace(" ", "").replace(",", "."))
    try:
        value = float(cleaned)
    except ValueError:
        return None, False
    # "%" has one meaning: division by 100. Every share-shaped question class
    # (`ReferenceSpec(op="share", ...)` in `sim/oracle/reference.py`) computes
    # its reference as a fraction of one, matching the contract's own worked
    # example of a bare decimal (0,3125). So 42% reads as 0.42, not 42 — this
    # is the notation's definition, not a guess between two readings.
    return (value / 100.0, True) if as_share else (value, True)


def _ids(raw: str) -> list[str]:
    text = _scalar(raw)
    if text is None:
        return []
    body = text.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    return [part.strip().strip('"').strip("'")
            for part in body.split(",") if part.strip()]


def _refused(raw: str) -> tuple[bool, bool]:
    """Parse the refusal flag. Returns ``(value, ok)``, same contract as
    ``_number``: ``ok`` is False for a spelling this parser does not
    recognise, so an unreadable ``refused: maybe`` cannot look like the
    ordinary ``refused: false``."""
    text = _scalar(raw)
    if text is None:
        return False, True
    word = text.lower()
    if word in _TRUE_WORDS:
        return True, True
    if word in _FALSE_WORDS:
        return False, True
    return False, False


def _split_fields(block: str) -> dict[str, str]:
    """Turn the block into ``key -> raw value text``.

    Handles two YAML shapes for a list-valued field: the flow style the
    contract shows (``ids: [a, b]``) and the block style a model reaches for
    just as often — ``ids:`` with nothing after the colon, followed by
    indented ``- item`` lines. The block style is folded into the same
    bracketed text ``_ids`` already parses, so there is exactly one place that
    understands list syntax.
    """
    fields: dict[str, str] = {}
    current_key: str | None = None
    list_items: list[str] = []

    def flush() -> None:
        nonlocal current_key, list_items
        if current_key is not None and list_items:
            fields[current_key] = "[" + ", ".join(list_items) + "]"
        current_key, list_items = None, []

    for line in block.splitlines():
        stripped = line.strip()
        if current_key is not None and stripped.startswith("-"):
            list_items.append(stripped[1:].strip())
            continue
        flush()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lstrip("-").strip().lower()
        if key not in _FIELD_NAMES:
            continue
        val = val.strip()
        if val:
            fields[key] = val
        else:
            current_key = key  # a block list may follow on later lines
    flush()
    return fields


def parse_answer(text: str) -> ParsedAnswer:
    """Read the last ``answer`` block, or say why there is none.

    The *last* block, not the first: a model that corrects itself emits two, and
    the correction is the answer. Returning an explicit ``error`` rather than an
    empty result is what lets the report separate "answered wrongly" from "did
    not answer in the required shape", which are different failures and, on
    memory arms, may occur at different rates.
    """
    blocks = _BLOCK.findall(text or "")
    if not blocks:
        return ParsedAnswer(present=False, error="no_answer_block")

    raw = _split_fields(blocks[-1])
    field_errors: dict[str, str] = {}

    value, value_ok = _number(raw.get("value", ""))
    if not value_ok:
        field_errors["value"] = "unparseable_number"

    refused, refused_ok = _refused(raw.get("refused", ""))
    if not refused_ok:
        field_errors["refused"] = "unparseable_bool"

    return ParsedAnswer(
        present=True,
        verdict=_scalar(raw.get("verdict", "")),
        ids=_ids(raw.get("ids", "")),
        value=value,
        refused=refused,
        reason=_scalar(raw.get("reason", "")),
        error=None,
        field_errors=field_errors,
    )
