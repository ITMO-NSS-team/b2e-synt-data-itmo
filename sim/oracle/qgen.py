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
from typing import Any, Callable, Sequence

from .labels import GoldLabels
from .reference import ReferenceSpec

#: Per-class targets. Asserted at generation time so the basket cannot drift
#: away from the document that describes it.
#:
#: Every class here is answerable *in principle* from the marts — see
#: ``MART_PATHS`` for where from, and the plan's Task 2 section for the
#: measurement that put five earlier classes out. Answerability is not a
#: quality bar, it is a validity one: a question whose reference quantity no
#: served column determines is not a hard question, it is a floor under every
#: arm equally, and 75 of 180 such questions would have spent almost half the
#: deterministic basket measuring nothing.
CLASS_COUNTS: dict[str, int] = {
    "count_by_grade": 20,
    "share_by_grade": 20,
    "count_heads": 15,
    "mean_by_unit": 15,
    "median_by_unit": 15,
    "top_n_competency": 15,
    "lookup_unit": 15,
    "lookup_grade": 15,
    "compare_competency": 15,
    "compare_grade": 15,
    "exists_senior": 20,
}

_FAMILY_OF: dict[str, str] = {
    "count_by_grade": "org", "share_by_grade": "org", "count_heads": "org",
    "mean_by_unit": "team_analysis", "median_by_unit": "team_analysis",
    "top_n_competency": "key_employees",
    "lookup_unit": "profile", "lookup_grade": "profile",
    "compare_competency": "compare_people", "compare_grade": "compare_people",
    "exists_senior": "org",
}

#: For every population field a generated reference is computed from, the
#: catalogue columns through which an agent can reach the same quantity.
#:
#: This map is the standing answer to "is this question answerable at all",
#: and it is data rather than prose so a test can check it against
#: ``catalog/snapshot.json``. Three fields the reference evaluator supports are
#: deliberately absent, and their absence is the reason five question classes
#: were replaced:
#:
#: * ``performance_pct`` — the snapshot-wide rank of ``perf_latent``. What the
#:   marts serve is ``estimation.performance``, an A-E ordinal mark drawn
#:   through a quantile model with a per-rater bias that
#:   ``b2e/gen/population.py::_ordinal_marks`` adds *on purpose* so that
#:   averaging the eight quarters cannot recover the latent. Measured on the
#:   800-person corpus: the served mark decides only 85% of random pairs and,
#:   where it decides, agrees with the latent ordering 75% of the time.
#: * ``potential_pct`` — the rank of ``truth['potential']``. No served column
#:   is a function of it alone; ``talent_pool`` is binary and ``career_status``
#:   is a four-way categorical, both mixing potential with tenure and ability.
#: * ``impact_pct`` — a composite whose weights exist only in
#:   ``sim/oracle/labels.py::IMPACT_WEIGHTS``. No mart publishes the weights or
#:   the snapshot-wide ranks it combines, so there is no quantity to compute.
MART_PATHS: dict[str, tuple[str, ...]] = {
    "grade_level": ("dm_core.employee_actual.grade_level",),
    "is_head": ("dm_core.employee_actual.position_flag_boss",),
    "unit_name": ("dm_core.employee_oshs.unit_name",),
    # The nine populated competency scores. Every other numeric column on this
    # mart is NULL for every row, so "the average of the competency scores" is
    # discoverable rather than a convention the question has to state — which
    # it states anyway, for the same reason the unit scope is stated.
    "competency_avg": (
        "dm_special.employee_competence_actual.personality_traits_group_score",
        "dm_special.employee_competence_actual.cognitive_features_group_score",
        "dm_special.employee_competence_actual.soft_skills_competency_score",
        "dm_special.employee_competence_actual.management_competency_score",
        "dm_special.employee_competence_actual.wide_context",
        "dm_special.employee_competence_actual.influence_scale",
        "dm_special.employee_competence_actual.reflection",
        "dm_special.employee_competence_actual.life_intelligenc",
        "dm_special.employee_competence_actual.extrinsic_motivation",
    ),
}

#: How a unit-scoped question's membership is reachable. Every unit-scoped class
#: evaluates with ``recursive=True``, so the agent needs the ancestor chain, not
#: just the person's own unit. ``dm_core.employee_actual`` carries it as fifteen
#: level columns; a person belongs to unit U's subtree exactly when U's name
#: appears among them.
#:
#: The ``_unit_name_main`` columns, not ``_unit_id_main``. Every unit-scoped
#: question names its unit in prose — "«{unit}»", never an id — because
#: ``spec.scope["unit_id"]`` is an index into the gold tree (`GoldLabels.tree`),
#: assigned when the corpus is built, and has nothing to do with the id space
#: these mart columns hold (they are strings like ``'1000000'`` from
#: ``b2e/gen``'s org generator). An agent has only the rendered name to filter
#: on, so that is the reachable column; the id columns are reachable by nothing
#: the agent is ever shown. Verified by replaying a generated question through
#: ``heimdall.engine.execute`` using only the named columns — see
#: ``tests/test_oracle_qgen.py::test_unit_scope_paths_reproduce_a_unit_scoped_answer_through_heimdall``.
UNIT_SCOPE_PATHS: tuple[str, ...] = tuple(
    f"dm_core.employee_actual.oshs_level_{level}_unit_name_main"
    for level in range(1, 16))

_MIN_UNIT_SIZE = 12

#: Said in every unit-scoped question, because the reference always evaluates
#: recursively and the text never said so. On the 800-person corpus 95 of 120
#: named units have no direct members at all — they are internal org nodes —
#: and 92 of 120 questions get a different answer under a direct-membership
#: reading. That is not a hard question, it is one sentence of unstated
#: convention worth tens of percentage points to whichever arm memorises it
#: first, against an expected between-arm effect of a few. The spec's
#: anti-leakage machinery protects against memorising facts; nothing protects
#: against memorising the scorer's conventions except stating them.
_SUBTREE = "включая подчинённые подразделения"

#: Said in every question about the competency average, for the same reason:
#: the quantity is the mean of a person's competency scores, and a question that
#: leaves the reader to guess which numbers are meant is scoring the guess.
_COMPETENCY_NOTE = ("Балл сотрудника — среднее по всем его оценкам "
                    "компетенций.")


def _h(seed: int, *parts: Any) -> int:
    """A stable integer from a seed and any tuple of parts."""
    raw = "|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _pick(items: Sequence[Any], seed: int, *parts: Any) -> Any:
    return items[_h(seed, *parts) % len(items)]


#: Retries budgeted per draw before a class is declared unable to fill without
#: colliding. Generous relative to the actual pool sizes (~80 eligible units,
#: 523 people, at most 20 draws per class) so it is never the limiting factor
#: on a real corpus; it exists to turn a starved pool into a loud failure
#: instead of a basket that silently comes up short of 180.
_MAX_DRAW_TRIES = 500


def _unique_draw(class_name: str, index: int, used: set[Any],
                 candidate: Callable[[int], Any],
                 key: Callable[[Any], Any] = lambda x: x,
                 valid: Callable[[Any], bool] = lambda x: True) -> Any:
    """Draw one value per question, retried until it does not repeat a value
    already used earlier in the same class and satisfies ``valid``.

    ``valid`` is how a class states the condition under which its question has
    exactly one defensible answer: a comparison needs the two people to differ
    on the compared field, a ranking needs its cut to be unambiguous. Rejecting
    at draw time is the only place that can be done — by the time a spec exists
    the question text has been written, and ``evaluate`` would happily compute
    one of several equally correct answers and mark the other ones wrong.

    ``_pick`` alone samples independently at every index — sampling *with*
    replacement from a finite pool. Two different indices in the same class
    landing on the same unit (or person, or unit+grade pair) is not a
    corner case on this corpus's pool sizes; it happens on every seed and
    produces two questions that are byte-identical in their Russian text. A
    duplicate does not add information to a basket asked once per agent
    configuration — it double-weights one draw and shrinks the effective
    sample size the experiment needs to resolve a few points of difference
    between configurations.

    Determinism survives because ``attempt`` is folded into the hash input
    on each retry rather than drawn from any RNG state — the same
    ``(seed, class, index, attempt)`` tuple always hashes to the same
    candidate, so two calls with the same seed retry down the exact same
    sequence and land on the exact same answer. Exhausting
    ``_MAX_DRAW_TRIES`` raises rather than returning a duplicate anyway: a
    pool too small to fill ``CLASS_COUNTS[class_name]`` without collisions
    is a corpus too small for this benchmark, and a basket that quietly
    shrank below 180 would be a silently changed experiment.
    """
    for attempt in range(_MAX_DRAW_TRIES):
        value = candidate(attempt)
        k = key(value)
        if k not in used and valid(value):
            used.add(k)
            return value
    raise ValueError(
        f"could not draw a unique, unambiguous value for {class_name!r} index "
        f"{index} after {_MAX_DRAW_TRIES} tries; the pool is too small for "
        f"CLASS_COUNTS[{class_name!r}]")


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


def _unambiguous_units(gold: GoldLabels) -> list[int]:
    """Eligible units whose rendered name names exactly one unit anywhere.

    A question is text, not an id: "«Управление операционных рисков»" is what
    reaches the agent, and ``reference.evaluate`` scores against exactly one
    ``unit_id``. If the org tree contains a second unit with that same
    rendered name, the question has more than one correct answer depending
    on which unit the agent (reasonably) assumes, and ``evaluate`` computes
    only one of them — a forced wrong answer regardless of how well the
    agent queried. On the 800-person test corpus 20 of 163 unit names are
    shared by 2-3 different unit ids (e.g. "Управление операционных рисков"
    is units 32, 54 *and* 82), and this is not a rare edge case: cross-
    evaluating every unit-scoped question this generator produced against
    every unit sharing its name found ~15-19% of a basket disagreeing across
    candidates.

    The name-collision count is taken over ``range(len(gold.tree))`` — every
    unit in the whole tree — deliberately, not over ``_eligible_units()``'s
    already-filtered list and not over ``set(gold.unit_of)``. This is the
    third time this module has been bitten by scoping a tree-wide property
    to a subset: an 8-person unit far too small to ever be drawn still makes
    an eligible 24-person unit's name ambiguous to the agent reading the
    question, because the agent has no way to know the small sibling was
    excluded from the generator's pool. Ambiguity is a property of the name
    across the *whole* org chart, not of the subset this generator happens
    to sample from.
    """
    name_counts: dict[str, int] = {}
    for u in range(len(gold.tree)):
        name = _unit_name(gold, u)
        name_counts[name] = name_counts.get(name, 0) + 1
    return [u for u in _eligible_units(gold) if name_counts[_unit_name(gold, u)] == 1]


def _top_is_strict(gold: GoldLabels, unit_id: int, n: int) -> bool:
    """Are the top ``n + 1`` competency averages in this unit all distinct?

    ``n + 1`` and not ``n``: the ranking is single-valued only if both the
    order *within* the answer and the cut *at its edge* are determined. Two
    people tied at rank n and n+1 make "the top n" an arbitrary choice between
    them, which ``evaluate`` resolves by snapshot row order — deterministic,
    and invisible to any agent.
    """
    values = sorted(
        (float(v) for v in gold.competency_avg[gold.members(int(unit_id))]),
        reverse=True)[:n + 1]
    return len(values) == n + 1 and len(set(values)) == n + 1


def _separates(gold: GoldLabels, field: str, a: str, b: str) -> bool:
    """Do these two people actually differ on the field being compared?"""
    rows = [gold.index_of(a), gold.index_of(b)]
    if any(r is None for r in rows):
        return False
    values = getattr(gold, field)
    return float(values[rows[0]]) != float(values[rows[1]])


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
    # Unit-scoped questions draw from the unambiguous pool, not the merely
    # eligible one: a unit big enough to give a non-degenerate count/median is
    # still unusable if its rendered name also belongs to another unit
    # somewhere in the tree, per `_unambiguous_units`'s docstring.
    units = _unambiguous_units(gold)
    if len(units) < 8:
        raise ValueError(
            f"snapshot has only {len(units)} units with >={_MIN_UNIT_SIZE} "
            "people and an unambiguous name; generation needs a larger "
            "or more diversely-named corpus")
    people = list(gold.person_id)
    out: list[GeneratedQuestion] = []

    # Dedup keys on the unit *name*, not the unit id: the org tree reuses
    # department names across branches (e.g. two different "Управление
    # операционных рисков" at different ids), so two distinct unit ids can
    # still render byte-identical Russian text. `_unique_draw`'s `key`
    # normalises to what actually determines the text.
    used_count_by_grade: set[tuple[str, int]] = set()
    for i in range(CLASS_COUNTS["count_by_grade"]):
        unit, grade = _unique_draw(
            "count_by_grade", i, used_count_by_grade,
            lambda attempt, i=i: (_pick(units, seed, "count_by_grade", i, attempt),
                                  10 + (_h(seed, "grade", i, attempt) % 6)),
            key=lambda t: (_unit_name(gold, t[0]), t[1]))
        out.append(_make(
            seed, "count_by_grade", i,
            f"Сколько сотрудников подразделения «{_unit_name(gold, unit)}», "
            f"{_SUBTREE}, имеют грейд {grade} или выше?",
            ReferenceSpec(op="count", field="grade_level",
                          scope={"unit_id": unit, "recursive": True},
                          predicate={"op": ">=", "value": grade}),
            (f"unit:{unit}",)))

    used_share_by_grade: set[tuple[str, int]] = set()
    for i in range(CLASS_COUNTS["share_by_grade"]):
        unit, grade = _unique_draw(
            "share_by_grade", i, used_share_by_grade,
            lambda attempt, i=i: (_pick(units, seed, "share_by_grade", i, attempt),
                                  12 + (_h(seed, "share_grade", i, attempt) % 4)),
            key=lambda t: (_unit_name(gold, t[0]), t[1]))
        out.append(_make(
            seed, "share_by_grade", i,
            f"Какая доля сотрудников подразделения «{_unit_name(gold, unit)}», "
            f"{_SUBTREE}, имеет грейд {grade} или выше? "
            f"Ответ — долей от единицы.",
            ReferenceSpec(op="share", field="grade_level",
                          scope={"unit_id": unit, "recursive": True},
                          predicate={"op": ">=", "value": grade}),
            (f"unit:{unit}",)))

    used_count_heads: set[str] = set()
    for i in range(CLASS_COUNTS["count_heads"]):
        unit = _unique_draw(
            "count_heads", i, used_count_heads,
            lambda attempt, i=i: _pick(units, seed, "count_heads", i, attempt),
            key=lambda u: _unit_name(gold, u))
        out.append(_make(
            seed, "count_heads", i,
            f"Сколько руководителей подразделений работает в подразделении "
            f"«{_unit_name(gold, unit)}», {_SUBTREE}?",
            ReferenceSpec(op="count", field="is_head",
                          scope={"unit_id": unit, "recursive": True},
                          predicate={"op": ">=", "value": 1}),
            (f"unit:{unit}",)))

    used_mean_by_unit: set[str] = set()
    for i in range(CLASS_COUNTS["mean_by_unit"]):
        unit = _unique_draw(
            "mean_by_unit", i, used_mean_by_unit,
            lambda attempt, i=i: _pick(units, seed, "mean_by_unit", i, attempt),
            key=lambda u: _unit_name(gold, u))
        out.append(_make(
            seed, "mean_by_unit", i,
            f"Каков средний балл компетенций сотрудников подразделения "
            f"«{_unit_name(gold, unit)}», {_SUBTREE}? {_COMPETENCY_NOTE}",
            ReferenceSpec(op="mean", field="competency_avg",
                          scope={"unit_id": unit, "recursive": True}),
            (f"unit:{unit}",)))

    used_median_by_unit: set[str] = set()
    for i in range(CLASS_COUNTS["median_by_unit"]):
        unit = _unique_draw(
            "median_by_unit", i, used_median_by_unit,
            lambda attempt, i=i: _pick(units, seed, "median_by_unit", i, attempt),
            key=lambda u: _unit_name(gold, u))
        out.append(_make(
            seed, "median_by_unit", i,
            f"Какова медиана грейда сотрудников подразделения "
            f"«{_unit_name(gold, unit)}», {_SUBTREE}?",
            ReferenceSpec(op="median", field="grade_level",
                          scope={"unit_id": unit, "recursive": True}),
            (f"unit:{unit}",)))

    # A ranking is only single-valued if its cut is. `evaluate` breaks ties by
    # snapshot row order, which is deterministic but invisible to the agent, so
    # a unit whose n-th and (n+1)-th competency averages are equal has several
    # equally right answers and exactly one of them scored. Requiring the top
    # n+1 values to be distinct removes the case rather than scoring it.
    used_top_n: set[tuple[str, int]] = set()
    for i in range(CLASS_COUNTS["top_n_competency"]):
        unit, n = _unique_draw(
            "top_n_competency", i, used_top_n,
            lambda attempt, i=i: (
                _pick(units, seed, "top_n_competency", i, attempt),
                3 + (_h(seed, "top_n_competency", "n", i, attempt) % 3)),
            key=lambda t: (_unit_name(gold, t[0]), t[1]),
            valid=lambda t: _top_is_strict(gold, t[0], t[1]))
        out.append(_make(
            seed, "top_n_competency", i,
            f"Назови {n} сотрудников подразделения «{_unit_name(gold, unit)}», "
            f"{_SUBTREE}, с наивысшим средним баллом компетенций. Перечисли их "
            f"person_id по убыванию балла. {_COMPETENCY_NOTE}",
            ReferenceSpec(op="top_n", field="competency_avg",
                          scope={"unit_id": unit, "recursive": True}, n=n),
            (f"unit:{unit}",)))

    used_lookup_unit: set[str] = set()
    for i in range(CLASS_COUNTS["lookup_unit"]):
        person = _unique_draw(
            "lookup_unit", i, used_lookup_unit,
            lambda attempt, i=i: _pick(people, seed, "lookup_unit", i, attempt))
        out.append(_make(
            seed, "lookup_unit", i,
            f"В каком подразделении работает сотрудник {person}? "
            f"Ответ — название подразделения.",
            ReferenceSpec(op="lookup", field="unit_name", refs=(person,)),
            (f"person:{person}",)))

    used_lookup_grade: set[str] = set()
    for i in range(CLASS_COUNTS["lookup_grade"]):
        person = _unique_draw(
            "lookup_grade", i, used_lookup_grade,
            lambda attempt, i=i: _pick(people, seed, "lookup_grade", i, attempt))
        out.append(_make(
            seed, "lookup_grade", i,
            f"Какой грейд у сотрудника {person}?",
            ReferenceSpec(op="lookup", field="grade_level", refs=(person,)),
            (f"person:{person}",)))

    # A comparison of two people who tie on the compared field has two right
    # answers and one scored one — `evaluate`'s `argmax` silently keeps the
    # first. Grades tie on 10.9% of random pairs on the test corpus, so this is
    # a routine draw rather than an edge case, and the pair is redrawn instead.
    for cls, fld, phrase, note in (
            ("compare_competency", "competency_avg",
             "средний балл компетенций", " " + _COMPETENCY_NOTE),
            ("compare_grade", "grade_level", "грейд", "")):
        used_compare: set[frozenset[str]] = set()
        for i in range(CLASS_COUNTS[cls]):
            def _draw_pair(attempt, cls=cls, i=i):
                a = _pick(people, seed, cls, "a", i, attempt)
                b = _pick(people, seed, cls, "b", i, attempt)
                if a == b:
                    b = people[(people.index(a) + 1) % len(people)]
                return a, b
            a, b = _unique_draw(
                cls, i, used_compare, _draw_pair,
                key=lambda pair: frozenset(pair),
                valid=lambda pair, fld=fld: _separates(gold, fld, *pair))
            out.append(_make(
                seed, cls, i,
                f"У кого выше {phrase} — {a} или {b}? "
                f"Ответ — person_id.{note}",
                ReferenceSpec(op="compare", field=fld, refs=(a, b)),
                (f"person:{a}", f"person:{b}")))

    used_exists_senior: set[tuple[str, int]] = set()
    for i in range(CLASS_COUNTS["exists_senior"]):
        unit, grade = _unique_draw(
            "exists_senior", i, used_exists_senior,
            lambda attempt, i=i: (_pick(units, seed, "exists_senior", i, attempt),
                                  15 + (_h(seed, "exists_grade", i, attempt) % 4)),
            key=lambda t: (_unit_name(gold, t[0]), t[1]))
        out.append(_make(
            seed, "exists_senior", i,
            f"Есть ли в подразделении «{_unit_name(gold, unit)}», {_SUBTREE}, "
            f"хотя бы один сотрудник грейда {grade} или выше? "
            f"Ответь да или нет.",
            ReferenceSpec(op="exists", field="grade_level",
                          scope={"unit_id": unit, "recursive": True},
                          predicate={"op": ">=", "value": grade}),
            (f"unit:{unit}",)))

    if len(out) != sum(CLASS_COUNTS.values()):
        raise RuntimeError(
            f"generated {len(out)}, declared {sum(CLASS_COUNTS.values())}")
    return tuple(out)
