"""The agent's structured tail, and how it is read.

Scoring free prose against a reference is a scorer whose disagreements are
partly its own — ``sim/oracle/labels.py`` makes the same argument about
returning dataclasses rather than sentences. So the agent appends a fenced
block, and correctness becomes a comparison rather than an interpretation.

The block is additive: the prose answer stays, because the presentation rubric
scores it and because a user-facing agent that emitted only YAML would not be
the agent under study.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sim.agent.prompt import ANSWER_CONTRACT  # noqa: F401  re-exported for tests

_BLOCK = re.compile(r"```answer\s*\n(.*?)```", re.S | re.I)
_NULLS = {"null", "none", "nil", "-", ""}


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    present: bool
    verdict: str | None = None
    ids: list[str] = field(default_factory=list)
    value: float | None = None
    refused: bool = False
    reason: str | None = None
    error: str | None = None


def _scalar(raw: str) -> str | None:
    text = raw.strip().strip('"').strip("'").strip()
    return None if text.lower() in _NULLS else text


def _number(raw: str) -> float | None:
    text = _scalar(raw)
    if text is None:
        return None
    # A Russian-language model writes 0,3125 as often as 0.3125, and thousands
    # separators arrive as spaces or non-breaking spaces.
    cleaned = (text.replace("\xa0", "").replace(" ", "")
                   .replace(" ", "").replace(",", "."))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _ids(raw: str) -> list[str]:
    text = _scalar(raw)
    if text is None:
        return []
    return [part.strip().strip('"').strip("'")
            for part in text.strip("[]").split(",") if part.strip()]


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

    fields: dict[str, str] = {}
    for line in blocks[-1].splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lstrip("-").strip().lower()
        if key in ("verdict", "ids", "value", "refused", "reason"):
            fields[key] = val

    refused_raw = (_scalar(fields.get("refused", "")) or "false").lower()
    return ParsedAnswer(
        present=True,
        verdict=_scalar(fields.get("verdict", "")),
        ids=_ids(fields.get("ids", "")),
        value=_number(fields.get("value", "")),
        refused=refused_raw in ("true", "да", "yes", "1"),
        reason=_scalar(fields.get("reason", "")),
        error=None,
    )
