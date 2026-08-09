"""Generating the deterministic half of the question basket.

The authored basket in ``sim/oracle/basket.py`` cannot carry this experiment
for two reasons its own docstring half-anticipates: 209 of its 265 questions
reach the model with literal ``{subject}`` slots because nothing in the agent
path calls ``Question.bind``, and its 30 ``gold_ref`` strings have no resolver.
Generating questions solves both at once — the binding happens here, and the
reference is a ``ReferenceSpec`` that is evaluable by construction.

Question text is Russian because everything the agent reads is Russian: the
system prompt, the refusal markers in ``sim/research/metrics.py``, the
injection canaries. An English question in a Russian corpus would be a
condition nobody declared.

Determinism is by explicit hashing rather than by ``random.seed``: a global
RNG makes generation order-dependent, and this function is called from a
driver that may generate several baskets in one process.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .labels import GoldLabels
from .reference import ReferenceSpec

#: Per-class targets, straight from the spec. Asserted at generation time so
#: the basket cannot drift away from the document that describes it.
CLASS_COUNTS: dict[str, int] = {
    "count_by_grade": 20,
    "share_by_grade": 20,
    "mean_by_unit": 15,
    "median_by_unit": 15,
    "top_n_potential": 15,
    "top_n_impact": 15,
    "lookup_unit": 15,
    "lookup_grade": 15,
    "compare_performance": 15,
    "compare_potential": 15,
    "exists_senior": 20,
}

_FAMILY_OF: dict[str, str] = {
    "count_by_grade": "org", "share_by_grade": "org",
    "mean_by_unit": "team_analysis", "median_by_unit": "team_analysis",
    "top_n_potential": "key_employees", "top_n_impact": "key_employees",
    "lookup_unit": "profile", "lookup_grade": "profile",
    "compare_performance": "compare_people", "compare_potential": "compare_people",
    "exists_senior": "org",
}

_MIN_UNIT_SIZE = 12


def _h(seed: int, *parts: Any) -> int:
    """A stable integer from a seed and any tuple of parts."""
    raw = "|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _pick(items: Sequence[Any], seed: int, *parts: Any) -> Any:
    return items[_h(seed, *parts) % len(items)]


def _eligible_units(gold: GoldLabels) -> list[int]:
    """Units big enough that a count or a median is not degenerate.

    Scans every unit in the org tree (``range(len(gold.tree))``), not just
    ``set(gold.unit_of)``. ``unit_of`` records each person's *direct* unit,
    which on a corpus this size is only the tree's leaves — on the 800-person
    test snapshot, 53 of 163 units, every one of them capping at 16 direct
    members because that is how ``org.build`` splits at the leaf level. A scan
    restricted to that set would still clear the ``len(units) < 8`` floor
    below (16 leaves qualify at ``_MIN_UNIT_SIZE=12``), so the shortage would
    never raise — it would instead silently pin every generated question to
    one of a handful of near-minimum-sized units, which is exactly the
    "median over a handful of people" degeneracy this threshold exists to
    prevent. ``gold.members(unit_id)`` defaults to ``recursive=True`` and so
    already returns a unit's whole subtree, which is what lets an internal
    (non-leaf) unit or the root qualify with a real population behind it.
    """
    return [u for u in range(len(gold.tree)) if len(gold.members(u)) >= _MIN_UNIT_SIZE]


def _unit_name(gold: GoldLabels, unit_id: int) -> str:
    return str(gold.tree.name[unit_id])


@dataclass(frozen=True, slots=True)
class GeneratedQuestion:
    id: str
    question_class: str
    family: str
    category: str
    text: str
    spec: ReferenceSpec
    entities: tuple[str, ...]


def _make(seed: int, question_class: str, index: int, text: str, spec: ReferenceSpec,
          entities: tuple[str, ...]) -> GeneratedQuestion:
    # `id` carries the seed, not just the class and the positional index.
    # Without it, `id` would be a pure function of (question_class, index) and
    # every seed would produce the exact same 180 ids in the exact same
    # order — indistinguishable from each other even though the *content*
    # (which unit, which grade bar, which people) differs per seed. That
    # would break the one property a generated id has to hold once it is
    # committed to the registry: two baskets from different seeds sitting in
    # the same registry must not collide on id.
    return GeneratedQuestion(
        id=f"gen.{question_class}.{seed}.{index:03d}",
        question_class=question_class,
        family=_FAMILY_OF[question_class],
        category="answerable",
        text=text,
        spec=spec,
        entities=entities,
    )


def generate(gold: GoldLabels, *, seed: int) -> tuple[GeneratedQuestion, ...]:
    """The 180 deterministic questions for one snapshot and one seed."""
    units = _eligible_units(gold)
    if len(units) < 8:
        raise ValueError(
            f"snapshot has only {len(units)} units with >={_MIN_UNIT_SIZE} "
            "people; generation needs a larger corpus")
    people = list(gold.person_id)
    out: list[GeneratedQuestion] = []

    for i in range(CLASS_COUNTS["count_by_grade"]):
        unit = _pick(units, seed, "count_by_grade", i)
        grade = 10 + (_h(seed, "grade", i) % 6)
        out.append(_make(
            seed, "count_by_grade", i,
            f"Сколько сотрудников в подразделении «{_unit_name(gold, unit)}» "
            f"имеют грейд {grade} или выше?",
            ReferenceSpec(op="count", field="grade_level",
                          scope={"unit_id": unit, "recursive": True},
                          predicate={"op": ">=", "value": grade}),
            (f"unit:{unit}",)))

    for i in range(CLASS_COUNTS["share_by_grade"]):
        unit = _pick(units, seed, "share_by_grade", i)
        grade = 12 + (_h(seed, "share_grade", i) % 4)
        out.append(_make(
            seed, "share_by_grade", i,
            f"Какая доля сотрудников подразделения «{_unit_name(gold, unit)}» "
            f"имеет грейд {grade} или выше? Ответ — долей от единицы.",
            ReferenceSpec(op="share", field="grade_level",
                          scope={"unit_id": unit, "recursive": True},
                          predicate={"op": ">=", "value": grade}),
            (f"unit:{unit}",)))

    for i in range(CLASS_COUNTS["mean_by_unit"]):
        unit = _pick(units, seed, "mean_by_unit", i)
        out.append(_make(
            seed, "mean_by_unit", i,
            f"Каков средний перцентиль результативности в подразделении "
            f"«{_unit_name(gold, unit)}»?",
            ReferenceSpec(op="mean", field="performance_pct",
                          scope={"unit_id": unit, "recursive": True}),
            (f"unit:{unit}",)))

    for i in range(CLASS_COUNTS["median_by_unit"]):
        unit = _pick(units, seed, "median_by_unit", i)
        out.append(_make(
            seed, "median_by_unit", i,
            f"Какова медиана среднего балла компетенций в подразделении "
            f"«{_unit_name(gold, unit)}»?",
            ReferenceSpec(op="median", field="competency_avg",
                          scope={"unit_id": unit, "recursive": True}),
            (f"unit:{unit}",)))

    for cls, fld, word in (("top_n_potential", "potential_pct", "потенциалу"),
                           ("top_n_impact", "impact_pct", "вкладу")):
        for i in range(CLASS_COUNTS[cls]):
            unit = _pick(units, seed, cls, i)
            n = 3 + (_h(seed, cls, "n", i) % 3)
            out.append(_make(
                seed, cls, i,
                f"Назови {n} сотрудников подразделения «{_unit_name(gold, unit)}» "
                f"с наивысшим показателем по {word}. Перечисли их person_id по "
                f"убыванию показателя.",
                ReferenceSpec(op="top_n", field=fld,
                              scope={"unit_id": unit, "recursive": True}, n=n),
                (f"unit:{unit}",)))

    for i in range(CLASS_COUNTS["lookup_unit"]):
        person = _pick(people, seed, "lookup_unit", i)
        out.append(_make(
            seed, "lookup_unit", i,
            f"В каком подразделении работает сотрудник {person}? "
            f"Ответ — название подразделения.",
            ReferenceSpec(op="lookup", field="unit_name", refs=(person,)),
            (f"person:{person}",)))

    for i in range(CLASS_COUNTS["lookup_grade"]):
        person = _pick(people, seed, "lookup_grade", i)
        out.append(_make(
            seed, "lookup_grade", i,
            f"Какой грейд у сотрудника {person}?",
            ReferenceSpec(op="lookup", field="grade_level", refs=(person,)),
            (f"person:{person}",)))

    for cls, fld, word in (("compare_performance", "performance_pct", "результативности"),
                           ("compare_potential", "potential_pct", "потенциалу")):
        for i in range(CLASS_COUNTS[cls]):
            a = _pick(people, seed, cls, "a", i)
            b = _pick(people, seed, cls, "b", i)
            if a == b:
                b = people[(people.index(a) + 1) % len(people)]
            out.append(_make(
                seed, cls, i,
                f"Кто выше по {word} — {a} или {b}? "
                f"Ответ — person_id победителя.",
                ReferenceSpec(op="compare", field=fld, refs=(a, b)),
                (f"person:{a}", f"person:{b}")))

    for i in range(CLASS_COUNTS["exists_senior"]):
        unit = _pick(units, seed, "exists_senior", i)
        grade = 15 + (_h(seed, "exists_grade", i) % 4)
        out.append(_make(
            seed, "exists_senior", i,
            f"Есть ли в подразделении «{_unit_name(gold, unit)}» хотя бы один "
            f"сотрудник грейда {grade} или выше? Ответь да или нет.",
            ReferenceSpec(op="exists", field="grade_level",
                          scope={"unit_id": unit, "recursive": True},
                          predicate={"op": ">=", "value": grade}),
            (f"unit:{unit}",)))

    if len(out) != sum(CLASS_COUNTS.values()):
        raise RuntimeError(
            f"generated {len(out)}, declared {sum(CLASS_COUNTS.values())}")
    return tuple(out)
