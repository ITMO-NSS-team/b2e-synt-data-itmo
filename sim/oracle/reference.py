"""Computable reference answers, expressed as data rather than as code.

A question's correct answer is a ``ReferenceSpec`` — an operation, a field, a
scope and an optional predicate — evaluated against the population. Two
consequences follow, and both are the point.

First, the reference never touches a mart. ``sim/oracle/labels.py`` already
takes this position for gold labels and states why: if the reference were
computed by querying the same projection the agent queries, a projection bug
would appear on both sides and cancel, scoring a pass. Here the agent goes
through Heimdall and the reference goes through ``truth/people.json``, so a
disagreement is always a real disagreement.

Second, because the spec is data, a generated question can be serialised,
committed to the registry alongside the run, and re-evaluated later against a
rebuilt snapshot. A reference expressed as a Python lambda could not be.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np

from .labels import GoldLabels, ROUND_DP

#: Operations a generated question may ask for. Closed set: an unknown op is a
#: generator bug, and a generator bug that silently returns ``None`` would be
#: scored as an agent failure on every question it produced.
OPS: tuple[str, ...] = (
    "count", "share", "mean", "median", "top_n", "lookup", "compare", "exists",
)

#: Population fields a question may be about. Every one is a per-person array on
#: ``GoldLabels``; ``unit_name`` is the single exception and is resolved through
#: the org tree.
FIELDS: tuple[str, ...] = (
    "grade_level", "performance_pct", "potential_pct", "competency_avg",
    "competency_pct", "impact_pct", "is_head", "unit_name",
)

_PREDICATES = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    op: str
    field: str
    scope: dict[str, Any] = dc_field(default_factory=dict)
    predicate: dict[str, Any] | None = None
    n: int | None = None
    refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Validated at construction, not inside evaluate()'s top_n branch,
        # because a ReferenceSpec is round-tripped through the run record: a
        # generated question is serialised to JSON and later deserialised
        # back into a ReferenceSpec to be re-evaluated against a rebuilt
        # snapshot. Construction is the one point both paths — a fresh spec
        # from the generator and one rehydrated from JSON — are guaranteed to
        # pass through, so it is the earliest point that reliably catches a
        # corrupt n before it can reach a slice expression and misbehave
        # silently (n=0 truncating to "top 1" via `or 1`; a negative n
        # reinterpreted as "drop the last k" via Python slice semantics).
        if self.n is not None and self.n < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}")


@dataclass(frozen=True, slots=True)
class Reference:
    kind: str                      # number | ids | verdict | boolean
    value: float | None = None
    ids: tuple[str, ...] = ()
    verdict: str | None = None


def _rows(spec: ReferenceSpec, gold: GoldLabels) -> np.ndarray:
    unit_id = spec.scope.get("unit_id")
    recursive = bool(spec.scope.get("recursive", True))
    return gold.members(None if unit_id is None else int(unit_id), recursive)


def _values(spec: ReferenceSpec, gold: GoldLabels, rows: np.ndarray) -> np.ndarray:
    if spec.field == "is_head":
        return gold.is_head[rows].astype(np.float64)
    if spec.field == "unit_name":
        raise ValueError("unit_name is not numeric; use op='lookup'")
    return np.asarray(getattr(gold, spec.field), dtype=np.float64)[rows]


def _filtered(spec: ReferenceSpec, gold: GoldLabels, rows: np.ndarray) -> np.ndarray:
    if spec.predicate is None:
        return rows
    op = spec.predicate["op"]
    if op not in _PREDICATES:
        raise ValueError(f"unknown predicate {op!r}")
    mask = _PREDICATES[op](_values(spec, gold, rows), float(spec.predicate["value"]))
    return rows[np.asarray(mask, dtype=bool)]


def _r(x: float) -> float:
    return round(float(x), ROUND_DP)


def evaluate(spec: ReferenceSpec, gold: GoldLabels) -> Reference:
    """The correct answer to one generated question.

    Raises rather than returning a sentinel on an unknown operation: a generator
    that emits a spec this cannot evaluate must fail at generation time, not
    turn into a question every agent answers wrongly.

    ``field`` is checked here for the same reason ``op`` is, and against the
    same closed set the module exports. It was previously checked nowhere: a
    typo'd field reached ``getattr(gold, ...)`` and surfaced as an
    ``AttributeError`` from numpy — a message naming neither the spec nor
    ``FIELDS`` — and only on the paths that happen to read values, so a
    predicate-free ``count`` swallowed it entirely.

    ``refs`` is checked for length rather than indexed on faith. A
    ``ReferenceSpec`` is round-tripped through JSON in the run record, so an
    empty tuple is a shape this function genuinely receives, and a bare
    ``IndexError`` from ``spec.refs[0]`` reads as a bug in this module rather
    than as the malformed spec it is.
    """
    if spec.op not in OPS:
        raise ValueError(f"unknown op {spec.op!r}, expected one of {OPS}")
    if spec.field not in FIELDS:
        raise ValueError(f"unknown field {spec.field!r}, expected one of {FIELDS}")

    if spec.op in ("count", "share"):
        rows = _rows(spec, gold)
        hit = _filtered(spec, gold, rows)
        if spec.op == "count":
            return Reference(kind="number", value=float(len(hit)))
        total = len(rows)
        return Reference(kind="number",
                         value=_r(len(hit) / total) if total else 0.0)

    if spec.op in ("mean", "median"):
        rows = _filtered(spec, gold, _rows(spec, gold))
        if len(rows) == 0:
            return Reference(kind="number", value=None)
        vals = _values(spec, gold, rows)
        agg = np.mean(vals) if spec.op == "mean" else np.median(vals)
        return Reference(kind="number", value=_r(agg))

    if spec.op == "top_n":
        rows = _filtered(spec, gold, _rows(spec, gold))
        vals = _values(spec, gold, rows)
        # Stable sort so ties break by row order, which is snapshot order, which
        # is deterministic. An unstable sort would make the reference depend on
        # numpy's build.
        # `spec.n` is validated in `ReferenceSpec.__post_init__` to be either
        # `None` or a positive integer, so the only defaulting left to do here
        # is None -> 1. `spec.n or 1` would look equivalent but is not: it
        # relies on Python truthiness, which treats 0 the same as None.
        limit = spec.n if spec.n is not None else 1
        order = rows[np.argsort(-vals, kind="stable")][:limit]
        return Reference(kind="ids",
                         ids=tuple(gold.person_id[int(r)] for r in order))

    if spec.op == "lookup":
        if len(spec.refs) < 1:
            raise ValueError("op='lookup' needs one entry in refs, got none")
        row = gold.index_of(spec.refs[0])
        if row is None:
            return Reference(kind="verdict", verdict=None)
        if spec.field == "unit_name":
            return Reference(kind="verdict",
                             verdict=str(gold.tree.name[gold.unit_of[row]]))
        return Reference(kind="number",
                         value=_r(_values(spec, gold, np.array([row]))[0]))

    if spec.op == "compare":
        if len(spec.refs) < 2:
            raise ValueError(
                f"op='compare' needs at least two entries in refs, got "
                f"{len(spec.refs)}")
        rows = [gold.index_of(r) for r in spec.refs]
        if any(r is None for r in rows):
            return Reference(kind="verdict", verdict=None)
        arr = np.array(rows, dtype=np.int64)
        winner = arr[int(np.argmax(_values(spec, gold, arr)))]
        return Reference(kind="verdict", verdict=gold.person_id[int(winner)])

    # exists
    rows = _filtered(spec, gold, _rows(spec, gold))
    return Reference(kind="boolean", value=float(len(rows) > 0))


#: Words a model may reasonably write into ``verdict`` for a yes/no question,
#: whose text says «Ответь да или нет» while the answer contract calls ``value``
#: «число». Deliberately kept here and not shared with
#: ``sim/research/answer.py``'s ``refused`` vocabulary: that one reads a
#: different field with a different meaning, and importing across would give
#: this module a dependency on the research package it has no other reason to
#: have.
_BOOLEAN_WORDS: dict[str, bool] = {
    "да": True, "yes": True, "true": True, "1": True,
    "нет": False, "no": False, "false": False, "0": False,
}

#: Punctuation a rendered name arrives wrapped in. Every unit-scoped question
#: class prints unit names inside guillemets, so a model answering
#: ``«Дирекция процессного офиса»`` is copying the format it was shown 105
#: times, not making a mistake.
_WRAPPERS = " \t\n«»\"'“”„`"


def _norm(text: str | None) -> str | None:
    """A verdict reduced to what it asserts, dropping how it was punctuated."""
    if text is None:
        return None
    stripped = text.strip().strip(_WRAPPERS).strip()
    return stripped.casefold() or None


def matches(ref: Reference, *, value: float | None,
            ids: list[str] | None, verdict: str | None) -> bool:
    """Does a parsed agent answer agree with the reference?

    Rankings compare order-sensitively. That is a real requirement rather than
    strictness for its own sake: "name the three highest-potential people" has a
    different correct answer from "name three high-potential people", and a
    set comparison would score the second when the first was asked. Numbers
    likewise stay exact at ``ROUND_DP``.

    Everything this function is lenient about is *format*, never content, and
    the distinction is the whole design. An answer-format convention the
    contract does not state is not a hard question — it is one sentence of
    learnable convention, and a single memory item that discovers it flips tens
    of questions at once. Against an expected between-arm effect of a few
    percentage points, the memory arms would show a spectacular learning curve
    that measures nothing about reflection. So: a yes/no question may be
    answered in the field the question told the model to use, a single named
    winner may arrive through either channel that can carry an identifier, and
    a name compares by what it names rather than by its quotation marks.
    """
    if ref.kind == "number":
        if value is None or ref.value is None:
            return value is None and ref.value is None
        return _r(value) == _r(ref.value)
    if ref.kind == "ids":
        return tuple(ids or ()) == ref.ids
    if ref.kind == "verdict":
        said = _norm(verdict)
        if said is None and ids is not None and len(ids) == 1:
            # Exactly one, never a shortlist: naming both candidates is not
            # naming the winner. Only when `verdict` is empty, so a model that
            # states a wrong winner is not rescued by a right id beside it —
            # that is hedging across two channels, not using the other one.
            said = _norm(ids[0])
        return said is not None and said == _norm(ref.verdict)
    if ref.kind == "boolean":
        if value is not None:
            return bool(value) == bool(ref.value)
        said = _BOOLEAN_WORDS.get(_norm(verdict) or "")
        return said is not None and said == bool(ref.value)
    raise ValueError(f"unknown reference kind {ref.kind!r}")
