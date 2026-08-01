"""Gold labels for the five task families of roadmap stage S8.

Why the label is computed from the population and never from the marts
----------------------------------------------------------------------
A mart is a *projection*. ``dm_core.employee_actual`` renames, rounds, buckets,
localises and occasionally corrupts what the generator knew — that is its job,
and ``b2e/traps.py`` makes some of the corruption deliberate. If the reference
answer were read back out of a mart, then every projection bug would be present
on both sides of the comparison and would cancel out: the agent would be scored
against the same defect it was fed, and the score would come out clean while the
system was wrong. Worse, the failure would be silent — a scorer that agrees with
a buggy mart looks exactly like a scorer that works.

So the rule for this module is absolute: **a label is a pure function of the
population.** Concretely, of two inputs only —

* ``truth/people.json`` — the latent factors and identifiers written straight out
  of ``b2e.gen.population``, never served through the API
  (``test_latent_factors_are_not_served`` enforces that), and
* the organisational tree, rebuilt from the snapshot seed via
  ``b2e.gen.org.build``, which is a pure function of ``(seed, headcount)``.

Nothing in this file opens a ``.col`` file, instantiates a ``Resolver``, or
imports ``b2e.gen.marts``. If a projection is wrong, the gold label stays right
and the disagreement shows up as a measured error — which is the entire point of
having a corpus with a known generative process.

The one apparent exception proves the rule. ``stale_key_employee_flag`` below
reproduces the corpus's ``is_key_employee`` column. It does so by re-deriving it
from the population and the seed — the same way the generator did — and never by
reading the column. It is labelled, in its own docstring and in every result it
appears in, as *not a gold label*: it exists so the deliberate disagreement
between last year's snapshot and a correct recomputation can be quantified
instead of merely asserted (see ``docs/research-agenda.md``, RQ1).

What the population does and does not carry
-------------------------------------------
``truth/people.json`` holds ten arrays: ``person_id``, ``employee_id``,
``ability``, ``potential``, ``perf_latent``, ``competency_avg``,
``attrition_risk``, ``is_head``, ``unit_id``, ``grade_level``. That is the whole
universe available to a label. Two consequences worth stating out loud rather
than discovering later:

* **There is no employment status in truth.** Roughly 12% of the snapshot is
  terminated (``ANNUAL_ATTRITION``), and the label frame therefore includes
  them. Every result carries ``frame="snapshot"`` so an S9 question can be worded
  to match ("across the snapshot", not "among active employees"). Narrowing the
  frame would require reading a mart, which is exactly what is forbidden.
* **There is no job family, tenure or engagement in truth.** A vacancy is
  therefore specified structurally — org scope, grade band, competency and
  potential bars — rather than by skill keywords. ``VacancySpec.requisition_id``
  can carry a reference to a row in ``recruitment.job_requisition_large`` for the
  question text, but it is *not read* and takes no part in scoring.

Why ranks and not z-scores
--------------------------
Every comparative statement here ("stronger than", "top decile", "above the bar
for the target grade") is expressed as a rank percentile of the reference frame,
not a standardised score. Three reasons. Ranks are invariant to the monotone
re-scalings the marts apply, so a question can be asked in mart units and scored
in population units. Ranks are exactly reproducible in integer arithmetic, so
the same snapshot yields byte-identical labels. And ranks degrade gracefully at
the tails, where a z-score on a factor model with clipped competencies does not.

Determinism
-----------
Same snapshot in, same labels out. There is no sampling anywhere. The only use
of a pseudo-random generator is ``stale_key_employee_flag``, which is seeded from
``manifest.json`` exactly as the build was, and is a reproduction of an existing
column rather than a fresh draw. All exported floats are rounded to
``ROUND_DP`` so two runs can be compared with ``==`` and not with a tolerance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, is_dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from b2e.gen import org
from b2e.gen.rng import key64, normal
from sim.registry import canonical_bytes, sha256_hex

# --------------------------------------------------------------------------
# Methodology constants.
#
# They live at module scope, and not inside the functions, because the S10
# rubric has to be able to cite the exact bar an answer was judged against.
# A threshold buried in an expression is a threshold that cannot be quoted to a
# grader, and an unquotable threshold makes a "wrong" verdict unarguable.
# --------------------------------------------------------------------------

#: Number of decimal places every exported float is rounded to. Chosen to sit
#: well inside the 4-decimal rounding already applied when ``truth`` was
#: written, so the label carries no precision the population does not have.
ROUND_DP = 4

#: Weights of the composite "contribution" score used by employee comparison,
#: vacancy fit and the key-employee methodology. Current delivery dominates,
#: potential is the second term because both promotion and retention decisions
#: are forward-looking, competency is a slower-moving third, and grade enters
#: only weakly — seniority is evidence of past decisions, not of present impact.
#: They sum to 1.0; the assertion below is a guard against an edit that forgets.
IMPACT_WEIGHTS: dict[str, float] = {
    "performance": 0.45,
    "potential": 0.25,
    "competency": 0.20,
    "seniority": 0.10,
}
assert abs(sum(IMPACT_WEIGHTS.values()) - 1.0) < 1e-12

#: Two composite scores closer than this are not a ranking. Expressed in
#: percentile units, so 0.03 means "less than three percentiles apart".
#: Comparison questions whose top two candidates fall inside this band are the
#: honest source of the S9 *ambiguous* category: the correct answer there is
#: "not distinguishable on the available evidence", and an agent that names a
#: winner anyway is wrong in a way worth measuring.
DECISIVE_MARGIN = 0.03

#: Key-employee methodology (the CORRECT one — contrast with the stale flag).
KEY_IMPACT_PCT = 0.90          # organisation-wide impact percentile
KEY_HEAD_IMPACT_PCT = 0.75     # heads carry structural criticality on top
KEY_SCARCE_GRADE = 17          # top ~3.7% of the grade pyramid
KEY_SCARCE_IMPACT_PCT = 0.60   # scarcity lowers, but does not remove, the bar

#: Readiness bars for the next grade step.
READINESS_THRESHOLDS: dict[str, float] = {
    "performance_pct": 0.60,
    "potential_pct": 0.65,
    "competency_vs_target_grade": 0.0,   # at or above the target-grade median
}

#: Minimum peers required before a grade-level benchmark is trusted. Below this
#: the median is noise, and the label says so instead of pretending.
MIN_BENCHMARK_PEERS = 12

#: Minimum members before a unit is admitted to the peer-unit ranking in team
#: analysis. A three-person team ranks first or last on one person's mood.
MIN_PEER_UNIT_SIZE = 5

#: Human-readable attrition risk bands, indexed by the integer in ``truth``.
ATTRITION_BANDS = ("low", "medium", "high")


# --------------------------------------------------------------------------
# Result shapes
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VacancySpec:
    """A vacancy expressed in terms the population can actually answer.

    Structural, not semantic, and on purpose. ``truth`` carries no job family
    and no skill vector, so a spec written in skill keywords could only be
    scored by reading a mart. What is left is still the substance of an internal
    move: where in the org, at what grade, over what bars, and who is excluded.

    ``requisition_id`` and ``title`` exist so an S9 question can quote a real
    row from ``recruitment.job_requisition_large`` and read naturally. Neither is
    consulted during matching — they are labels on the question, not inputs to
    the answer.
    """

    title: str = ""
    requisition_id: str = ""
    #: Org scope. ``None`` searches the whole snapshot.
    unit_id: int | None = None
    #: Whether ``unit_id`` means that unit alone or the whole subtree below it.
    recursive: bool = True
    grade_min: int = 6
    grade_max: int = 20
    #: A candidate one grade below ``grade_min`` is a promotion candidate rather
    #: than a lateral move: admitted, but at a discounted grade fit.
    allow_stretch: bool = True
    min_competency: float = 0.0
    min_potential_pct: float = 0.0
    min_performance_pct: float = 0.0
    exclude_high_attrition: bool = False
    exclude_heads: bool = False
    heads_only: bool = False
    top_k: int = 10


@dataclass(frozen=True, slots=True)
class Comparison:
    family: str
    frame: str
    requested: list[str]
    resolved: list[str]
    unknown_refs: list[str]
    ranking: list[dict[str, Any]]
    winner: str | None
    margin: float | None
    decisive: bool
    method: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VacancyMatch:
    family: str
    frame: str
    vacancy: dict[str, Any]
    pool_size: int
    eligible: int
    feasible: bool
    shortlist: list[dict[str, Any]]
    rejected_by_reason: dict[str, int]
    method: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Readiness:
    family: str
    frame: str
    person_id: str | None
    employee_id: int | None
    unknown_ref: str | None
    current_grade: int | None
    target_grade: int | None
    verdict: str
    readiness_score: float | None
    criteria: list[dict[str, Any]]
    benchmark: dict[str, Any]
    manager: dict[str, Any]
    method: dict[str, Any]


@dataclass(frozen=True, slots=True)
class KeyEmployees:
    family: str
    frame: str
    scope: dict[str, Any]
    population: int
    key_person_ids: list[str]
    key_at_risk_person_ids: list[str]
    people: list[dict[str, Any]]
    stale_flag_disagreement: dict[str, Any]
    method: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TeamProfile:
    family: str
    frame: str
    unit: dict[str, Any]
    headcount: dict[str, Any]
    head: dict[str, Any]
    grades: dict[str, Any]
    performance: dict[str, Any]
    competency: dict[str, Any]
    attrition: dict[str, Any]
    key_people: dict[str, Any]
    succession: dict[str, Any]
    peer_ranking: dict[str, Any]
    sub_units: list[dict[str, Any]]
    method: dict[str, Any]


def as_dict(result: Any) -> dict[str, Any]:
    """Dataclass result → plain JSON-able dict, for the scorer and for traces."""
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    raise TypeError(f"not an oracle result: {type(result)!r}")


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------

def _rank_pct(values: np.ndarray) -> np.ndarray:
    """Rank percentile in ``(0, 1)``, ties averaged.

    Vectorised rather than looped because it runs over the whole snapshot, which
    is 294 000 rows in the full corpus; and mid-rank rather than
    ``count_less/n`` because the grade pyramid produces large exact ties, and a
    convention that puts every tied person at the *bottom* of their tie block
    would make "top decile of grade" mean something different from what a human
    would read it to mean.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    sorted_v = v[order]
    starts_new = np.r_[True, sorted_v[1:] != sorted_v[:-1]]
    group = np.cumsum(starts_new) - 1
    counts = np.bincount(group)
    block_start = np.r_[0, np.cumsum(counts)[:-1]]
    mid_rank = block_start + (counts - 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = mid_rank[group]
    return (ranks + 0.5) / n


def _r(x: Any) -> Any:
    """Round for export. Keeps label files diffable and comparisons exact."""
    if x is None:
        return None
    return round(float(x), ROUND_DP)


def _pct_list(values: np.ndarray, qs: Sequence[float]) -> list[float]:
    if values.size == 0:
        return [0.0 for _ in qs]
    return [_r(v) for v in np.percentile(values, [q * 100 for q in qs])]


# --------------------------------------------------------------------------
# Org tree recovery
# --------------------------------------------------------------------------

_TREE_CACHE: dict[tuple[int, int, int], org.OrgTree] = {}


def rebuild_org_tree(seed: int, n_people: int, n_units: int) -> org.OrgTree:
    """Rebuild the exact tree the snapshot was generated with.

    ``org.build`` takes the *requested* headcount, but the tree it produces has
    a different *actual* headcount: the recursive split rounds, and
    ``assign_people`` honestly returns the sum of leaf headcounts rather than
    padding to a round number. ``manifest.json`` records only the actual figure,
    so the requested one has to be recovered — and it genuinely has to be
    recovered, not guessed. Building with the actual headcount produces a
    *smaller, different* tree (2 494 people in 503 units where the corpus has
    2 741 in 540), and a label computed on that tree would silently describe a
    different organisation.

    The recovery is a monotone search on the requested headcount followed by an
    exact check on both invariants the manifest does record — people and units.
    If the check cannot be satisfied the function raises, because a nearly-right
    org chart is worse than no org chart: it produces plausible team answers
    about teams that do not exist.
    """
    cache_key = (int(seed), int(n_people), int(n_units))
    hit = _TREE_CACHE.get(cache_key)
    if hit is not None:
        return hit

    def actual(target: int) -> tuple[org.OrgTree, int, int]:
        tree = org.build(seed, target)
        return tree, int(tree.headcount.sum()), len(tree)

    def matches(tree: org.OrgTree, people: int, units: int) -> bool:
        return people == n_people and units == n_units

    lo, hi = max(n_people, 1), max(n_people * 2, n_people + 128)
    best: org.OrgTree | None = None
    while lo <= hi:                      # smallest target reaching n_people
        mid = (lo + hi) // 2
        tree, people, units = actual(mid)
        if matches(tree, people, units):
            best = tree
            break
        if people < n_people:
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        # The split jitter is not strictly monotone in the requested headcount,
        # so the bisection can land beside the answer rather than on it. A short
        # local sweep costs milliseconds and removes the failure mode entirely.
        centre = min(max(lo, n_people), n_people * 2)
        for offset in range(0, 257):
            for target in {centre - offset, centre + offset}:
                if target < 1:
                    continue
                tree, people, units = actual(target)
                if matches(tree, people, units):
                    best = tree
                    break
            if best is not None:
                break

    if best is None:
        raise RuntimeError(
            f"cannot reproduce the org tree for seed={seed}: no requested "
            f"headcount yields {n_people} people in {n_units} units. Record the "
            f"requested headcount in manifest.json and pass it explicitly."
        )
    _TREE_CACHE[cache_key] = best
    return best


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------

class GoldLabels:
    """Reference answers for one snapshot.

    Constructed once and reused: the population arrays are a few megabytes, the
    derived percentile vectors are computed lazily, and the org tree costs a
    handful of ``org.build`` calls to recover. Instantiating this per question
    would make an S9 basket run of a few hundred questions spend most of its
    time rebuilding an org chart that cannot change.

    Every public method returns a frozen dataclass of identifiers, ranks and
    numbers. None of them returns prose. That is not stylistic: a scorer that
    has to parse a sentence is a scorer whose disagreements with the agent are
    partly its own, and S10 splits deterministic checks from rubric judgement
    precisely along this line.
    """

    def __init__(self, snapshot_root: str | Path,
                 *, requested_headcount: int | None = None) -> None:
        self.root = Path(snapshot_root)
        truth = json.loads((self.root / "truth" / "people.json").read_text("utf-8"))
        manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))

        self.person_id: list[str] = [str(x) for x in truth["person_id"]]
        self.employee_id: list[int] = [int(x) for x in truth["employee_id"]]
        self.ability = np.asarray(truth["ability"], dtype=np.float64)
        self.potential = np.asarray(truth["potential"], dtype=np.float64)
        self.perf_latent = np.asarray(truth["perf_latent"], dtype=np.float64)
        self.competency_avg = np.asarray(truth["competency_avg"], dtype=np.float64)
        self.attrition_risk = np.asarray(truth["attrition_risk"], dtype=np.int64)
        self.is_head = np.asarray(truth["is_head"], dtype=np.int64).astype(bool)
        self.unit_of = np.asarray(truth["unit_id"], dtype=np.int64)
        self.grade_level = np.asarray(truth["grade_level"], dtype=np.int64)

        self.n = len(self.person_id)
        self.seed = int(manifest["seed"])
        self.snapshot_id = str(manifest["snapshot_id"])
        self.as_of = str(manifest.get("as_of", ""))
        self.traps = tuple(manifest.get("traps", ()))
        self.n_units = int(manifest.get("units", 0))

        if requested_headcount is not None:
            self.tree = org.build(self.seed, int(requested_headcount))
            if int(self.tree.headcount.sum()) != self.n:
                raise RuntimeError(
                    f"requested_headcount={requested_headcount} yields "
                    f"{int(self.tree.headcount.sum())} people, snapshot has {self.n}")
        else:
            self.tree = rebuild_org_tree(self.seed, self.n, self.n_units)

        self._by_person = {p: i for i, p in enumerate(self.person_id)}
        self._by_employee = {str(e): i for i, e in enumerate(self.employee_id)}

    # ------------------------------------------------------------- identity

    def fingerprint(self) -> str:
        """Hash of what the labels depend on, for the run record.

        Deliberately covers the methodology constants as well as the snapshot:
        changing ``KEY_IMPACT_PCT`` changes every key-employee answer, and a run
        recorded before the change must not be silently comparable to one after.
        """
        return sha256_hex(canonical_bytes({
            "snapshot_id": self.snapshot_id,
            "seed": self.seed,
            "people": self.n,
            "units": len(self.tree),
            "impact_weights": IMPACT_WEIGHTS,
            "decisive_margin": DECISIVE_MARGIN,
            "key": [KEY_IMPACT_PCT, KEY_HEAD_IMPACT_PCT, KEY_SCARCE_GRADE,
                    KEY_SCARCE_IMPACT_PCT],
            "readiness": READINESS_THRESHOLDS,
            "round_dp": ROUND_DP,
        }))

    def index_of(self, ref: str | int) -> int | None:
        """Resolve a ``person_id`` or ``employee_id`` to a row, or ``None``.

        Returning ``None`` rather than raising is what makes the S9 *missing
        data* category expressible: a question about a person who is not in the
        snapshot has a correct answer, and that answer is a refusal.
        """
        key = str(ref)
        idx = self._by_person.get(key)
        if idx is None:
            idx = self._by_employee.get(key)
        return idx

    # ------------------------------------------------- derived population views

    @cached_property
    def performance_pct(self) -> np.ndarray:
        return _rank_pct(self.perf_latent)

    @cached_property
    def potential_pct(self) -> np.ndarray:
        return _rank_pct(self.potential)

    @cached_property
    def competency_pct(self) -> np.ndarray:
        return _rank_pct(self.competency_avg)

    @cached_property
    def seniority_pct(self) -> np.ndarray:
        return _rank_pct(self.grade_level.astype(np.float64))

    @cached_property
    def impact(self) -> np.ndarray:
        """Composite contribution score in ``[0, 1]``, snapshot-relative.

        Computed once over the whole snapshot rather than per query, because the
        alternative — re-ranking inside whatever subset a question happens to
        mention — would make the same person "top decile" in one question and
        "median" in another, and no consistent gold label could exist.
        """
        w = IMPACT_WEIGHTS
        return (w["performance"] * self.performance_pct
                + w["potential"] * self.potential_pct
                + w["competency"] * self.competency_pct
                + w["seniority"] * self.seniority_pct)

    @cached_property
    def impact_pct(self) -> np.ndarray:
        return _rank_pct(self.impact)

    @cached_property
    def _children(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for node, parent in enumerate(self.tree.parent):
            if parent >= 0:
                out.setdefault(int(parent), []).append(node)
        return out

    @cached_property
    def _rows_by_unit(self) -> dict[int, np.ndarray]:
        """unit_id → row indices of its *direct* members, built in one pass.

        The obvious implementation — ``np.isin(unit_of, subtree)`` per query —
        is O(people) every time it is called, and team analysis calls it once
        per peer unit. On the full corpus that is 32 500 passes over 294 000
        rows for a single question. One grouping pass up front removes the whole
        class of problem.
        """
        order = np.argsort(self.unit_of, kind="stable")
        sorted_units = self.unit_of[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_units[1:] != sorted_units[:-1]])
        edges = np.r_[boundaries, sorted_units.size]
        return {int(sorted_units[boundaries[i]]): np.sort(order[edges[i]:edges[i + 1]])
                for i in range(boundaries.size)}

    @cached_property
    def _ancestor_of_unit(self) -> np.ndarray:
        """``unit × level`` → the ancestor unit at that level, or -1.

        Reused from ``OrgTree.ancestors_matrix`` rather than re-walked: peer
        ranking needs "which unit at level L does this person roll up to" for
        every person at once, and a per-node ``path()`` walk is the exact
        pattern that method exists to replace.
        """
        return self.tree.ancestors_matrix()

    @cached_property
    def _competency_median_by_grade(self) -> dict[int, tuple[float, int]]:
        """grade → (median competency, incumbent count). Computed once.

        Readiness is asked for every member of a unit during team analysis, and
        each call needs a target-grade benchmark. Recomputing the median from a
        boolean mask each time makes team analysis quadratic in headcount for no
        reason — the benchmark is a property of the snapshot, not of the query.
        """
        out: dict[int, tuple[float, int]] = {}
        for grade in np.unique(self.grade_level):
            rows = np.flatnonzero(self.grade_level == int(grade))
            out[int(grade)] = (float(np.median(self.competency_avg[rows])), int(rows.size))
        return out

    def _benchmark_competency(self, target: int) -> tuple[float, int, str]:
        median, count = self._competency_median_by_grade.get(target, (float("nan"), 0))
        if count >= MIN_BENCHMARK_PEERS:
            return median, count, f"grade_level == {target}"
        rows = np.flatnonzero(self.grade_level >= target)
        if rows.size == 0:
            return float("nan"), 0, f"grade_level >= {target} (no incumbents)"
        return (float(np.median(self.competency_avg[rows])), int(rows.size),
                f"grade_level >= {target} (thin band at {target})")

    @cached_property
    def _head_of_unit(self) -> dict[int, int]:
        """unit_id → row index of the person flagged ``is_head`` in that unit.

        Derived from the population rather than from ``tree.head_person``,
        which ``org.build`` leaves empty: head appointment happens in
        ``population._appoint_heads``, and its outcome reaches us through the
        ``is_head`` column of ``truth``. Rebuilding the appointment here would
        duplicate that logic and could drift from it.
        """
        out: dict[int, int] = {}
        for row in np.flatnonzero(self.is_head):
            out.setdefault(int(self.unit_of[row]), int(row))
        return out

    def subtree(self, unit_id: int) -> list[int]:
        """A unit and every unit beneath it, in stable order."""
        seen: list[int] = []
        stack = [int(unit_id)]
        mark: set[int] = set()
        while stack:
            node = stack.pop()
            if node in mark:
                continue
            mark.add(node)
            seen.append(node)
            stack.extend(self._children.get(node, ()))
        return sorted(seen)

    def members(self, unit_id: int | None, recursive: bool = True) -> np.ndarray:
        """Row indices of the people in a unit (optionally its whole subtree)."""
        if unit_id is None:
            return np.arange(self.n, dtype=np.int64)
        units = self.subtree(unit_id) if recursive else [int(unit_id)]
        parts = [self._rows_by_unit[u] for u in units if u in self._rows_by_unit]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.sort(np.concatenate(parts))

    def _level_rollup(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-unit (headcount, mean performance percentile) at one tree level.

        Every person is attributed to their ancestor at ``level``, so a unit's
        figure always covers its whole subtree. That is the only comparison that
        means anything: a department's direct members are its management layer,
        and ranking departments by the strength of their management layer would
        answer a question nobody asked.
        """
        ancestor = self._ancestor_of_unit[self.unit_of, level - 1]
        n_units = len(self.tree)
        counts = np.bincount(ancestor[ancestor >= 0], minlength=n_units)
        totals = np.bincount(ancestor[ancestor >= 0],
                             weights=self.performance_pct[ancestor >= 0],
                             minlength=n_units)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        return counts, means

    def _peer_rank(self, unit_id: int, level: int) -> tuple[int | None, int, float | None]:
        counts, means = self._level_rollup(level)
        eligible = np.flatnonzero((self.tree.level == level)
                                  & (counts >= MIN_PEER_UNIT_SIZE))
        if eligible.size == 0:
            return None, 0, None
        order = sorted(eligible.tolist(), key=lambda node: (-float(means[node]), node))
        value = _r(means[unit_id]) if counts[unit_id] > 0 else None
        rank = order.index(unit_id) + 1 if unit_id in order else None
        return rank, int(eligible.size), value

    def unit_path(self, unit_id: int) -> list[str]:
        return [str(self.tree.name[node]) for node in self.tree.path(int(unit_id))]

    # ------------------------------------------------------------- card

    def card(self, row: int) -> dict[str, Any]:
        """The per-person block every family reuses. One shape, one meaning."""
        row = int(row)
        return {
            "person_id": self.person_id[row],
            "employee_id": self.employee_id[row],
            "unit_id": int(self.unit_of[row]),
            "unit_name": str(self.tree.name[self.unit_of[row]]),
            "grade_level": int(self.grade_level[row]),
            "is_head": bool(self.is_head[row]),
            "attrition_risk": ATTRITION_BANDS[int(self.attrition_risk[row])],
            "performance_pct": _r(self.performance_pct[row]),
            "potential_pct": _r(self.potential_pct[row]),
            "competency_avg": _r(self.competency_avg[row]),
            "competency_pct": _r(self.competency_pct[row]),
            "impact": _r(self.impact[row]),
            "impact_pct": _r(self.impact_pct[row]),
        }

    # ==================================================================
    # Family 1 — comparing employees
    # ==================================================================

    def compare(self, refs: Iterable[str | int]) -> Comparison:
        """Rank a named set of people against each other.

        The answer is a *ranking with margins*, not a winner. A question basket
        that only ever asks "who is stronger" cannot distinguish an agent that
        knows the two are indistinguishable from one that guesses correctly half
        the time, so the label reports the composite gap and a ``decisive`` flag
        derived from ``DECISIVE_MARGIN``. When ``decisive`` is false the correct
        answer is "not separable", and ``winner`` is still reported so a scorer
        can measure over-confidence rather than having to infer it.

        Unresolvable references are returned in ``unknown_refs`` instead of
        being dropped: a comparison against a person who does not exist is an
        answerability question, and silently comparing the rest would score an
        agent's hallucination as a partial success.
        """
        requested = [str(r) for r in refs]
        rows: list[int] = []
        unknown: list[str] = []
        for ref in requested:
            idx = self.index_of(ref)
            if idx is None or idx in rows:
                if idx is None:
                    unknown.append(ref)
                continue
            rows.append(int(idx))

        # Deterministic order: composite descending, person_id ascending. The
        # tie-break must be on a stable identifier and not on row order, or two
        # equally scored people would swap places with the corpus size.
        rows.sort(key=lambda r: (-float(self.impact[r]), self.person_id[r]))

        dims = ("performance_pct", "potential_pct", "competency_pct", "impact")
        cards = [self.card(row) for row in rows]
        # Per-dimension ranks are competition ranks broken by person_id, so the
        # answer to "who is strongest on competency" is a single id even when the
        # rounded percentiles collide. Ranking on the rounded, exported value
        # rather than on the raw float is intentional: the scorer only ever sees
        # the exported number, and a rank it cannot re-derive from the payload
        # is a rank it cannot check.
        ranking: list[dict[str, Any]] = []
        for position, (row, entry) in enumerate(zip(rows, cards), start=1):
            entry["rank"] = position
            entry["dimension_ranks"] = {
                dim: 1 + sum(1 for other in cards
                             if (other[dim] > entry[dim])
                             or (other[dim] == entry[dim]
                                 and other["person_id"] < entry["person_id"]))
                for dim in dims
            }
            ranking.append(entry)

        margin = None
        if len(rows) >= 2:
            margin = _r(float(self.impact[rows[0]]) - float(self.impact[rows[1]]))
        winner = self.person_id[rows[0]] if rows else None
        decisive = bool(margin is not None and margin >= DECISIVE_MARGIN)

        return Comparison(
            family="compare_employees",
            frame="snapshot",
            requested=requested,
            resolved=[self.person_id[r] for r in rows],
            unknown_refs=unknown,
            ranking=ranking,
            winner=winner,
            margin=margin,
            decisive=decisive,
            method={
                "composite": "impact",
                "weights": dict(IMPACT_WEIGHTS),
                "decisive_margin": DECISIVE_MARGIN,
                "tie_break": "person_id ascending",
                "note": ("percentiles are snapshot-wide, so a person's score does "
                         "not depend on who else was named in the question"),
            },
        )

    # ==================================================================
    # Family 2 — matching to a vacancy
    # ==================================================================

    def match_vacancy(self, spec: VacancySpec) -> VacancyMatch:
        """Shortlist internal candidates for a structurally specified vacancy.

        Two-phase on purpose: hard filters first, then a fit score over what
        survives. Folding the bars into the score would let a strong performer
        two grades below the band outrank an eligible one, which is not how an
        internal move works and would make the gold answer indefensible in the
        rubric.

        ``rejected_by_reason`` is part of the answer, not diagnostics. "There is
        no suitable candidate" is a correct answer that S9 must be able to ask
        for, and it is only checkable if the label says how many people were
        looked at and why each bucket was dropped.
        """
        pool = self.members(spec.unit_id, spec.recursive)
        reasons: dict[str, int] = {}

        def drop(mask: np.ndarray, reason: str, remaining: np.ndarray) -> np.ndarray:
            dropped = int((~mask).sum())
            if dropped:
                reasons[reason] = reasons.get(reason, 0) + dropped
            return remaining[mask]

        rows = pool
        floor = spec.grade_min - 1 if spec.allow_stretch else spec.grade_min
        g = self.grade_level[rows]
        rows = drop((g >= floor) & (g <= spec.grade_max), "grade_out_of_band", rows)
        rows = drop(self.competency_avg[rows] >= spec.min_competency,
                    "competency_below_bar", rows)
        rows = drop(self.potential_pct[rows] >= spec.min_potential_pct,
                    "potential_below_bar", rows)
        rows = drop(self.performance_pct[rows] >= spec.min_performance_pct,
                    "performance_below_bar", rows)
        if spec.exclude_high_attrition:
            rows = drop(self.attrition_risk[rows] < 2, "high_attrition_risk", rows)
        if spec.exclude_heads:
            rows = drop(~self.is_head[rows], "is_a_unit_head", rows)
        if spec.heads_only:
            rows = drop(self.is_head[rows], "not_a_unit_head", rows)

        # Grade fit: full credit inside the band, discounted for a stretch move.
        # The discount is what stops the label from recommending a promotion
        # whenever a slightly stronger junior exists — the point of a grade band
        # is that it is a constraint, not a preference.
        scored: list[dict[str, Any]] = []
        for row in rows:
            row = int(row)
            grade = int(self.grade_level[row])
            stretch = grade < spec.grade_min
            grade_fit = 0.75 if stretch else 1.0
            fit = (IMPACT_WEIGHTS["performance"] * self.performance_pct[row]
                   + IMPACT_WEIGHTS["potential"] * self.potential_pct[row]
                   + IMPACT_WEIGHTS["competency"] * self.competency_pct[row]
                   + IMPACT_WEIGHTS["seniority"] * grade_fit)
            entry = self.card(row)
            entry["fit"] = _r(fit)
            entry["grade_fit"] = grade_fit
            entry["is_stretch"] = stretch
            scored.append(entry)

        scored.sort(key=lambda e: (-e["fit"], e["person_id"]))
        for position, entry in enumerate(scored[:max(spec.top_k, 0)], start=1):
            entry["rank"] = position

        return VacancyMatch(
            family="match_to_vacancy",
            frame="snapshot",
            vacancy=asdict(spec),
            pool_size=int(pool.size),
            eligible=len(scored),
            feasible=bool(scored),
            shortlist=scored[:max(spec.top_k, 0)],
            rejected_by_reason=dict(sorted(reasons.items())),
            method={
                "phases": ["hard filters", "fit score over survivors"],
                "weights": dict(IMPACT_WEIGHTS),
                "stretch_grade_fit": 0.75,
                "tie_break": "person_id ascending",
                "note": ("title and requisition_id are carried for question "
                         "wording only and take no part in scoring"),
            },
        )

    # ==================================================================
    # Family 3 — readiness for a career step
    # ==================================================================

    def readiness(self, ref: str | int, target_grade: int | None = None) -> Readiness:
        """Is this person ready for the next grade, and on what evidence.

        The competency bar is *relative to the target grade*, not absolute. An
        absolute bar makes readiness a restatement of seniority — everyone at
        G16 clears any fixed number — and the question stops discriminating. The
        median incumbent at the target grade is the honest benchmark: it is what
        "could do the job today" means.

        The verdict deliberately has a middle value. Forcing ready/not-ready
        collapses the most common real case (delivers today, needs breadth) into
        one of the extremes, and an agent that names it correctly should not be
        scored the same as one that does not.
        """
        idx = self.index_of(ref)
        if idx is None:
            return Readiness(
                family="readiness_for_step", frame="snapshot", person_id=None,
                employee_id=None, unknown_ref=str(ref), current_grade=None,
                target_grade=None, verdict="unknown_person", readiness_score=None,
                criteria=[], benchmark={}, manager={},
                method={"note": "reference does not resolve in this snapshot; "
                                "the correct answer is a refusal"},
            )

        row = int(idx)
        current = int(self.grade_level[row])
        target = int(target_grade) if target_grade is not None else current + 1

        bench_comp, peer_count, benchmark_frame = self._benchmark_competency(target)

        perf = float(self.performance_pct[row])
        pot = float(self.potential_pct[row])
        comp_gap = float(self.competency_avg[row]) - bench_comp if peer_count else 0.0

        criteria = [
            {"name": "performance_pct", "value": _r(perf),
             "threshold": READINESS_THRESHOLDS["performance_pct"],
             "passed": bool(perf >= READINESS_THRESHOLDS["performance_pct"])},
            {"name": "potential_pct", "value": _r(pot),
             "threshold": READINESS_THRESHOLDS["potential_pct"],
             "passed": bool(pot >= READINESS_THRESHOLDS["potential_pct"])},
            {"name": "competency_vs_target_grade", "value": _r(comp_gap),
             "threshold": READINESS_THRESHOLDS["competency_vs_target_grade"],
             "passed": bool(peer_count > 0
                            and comp_gap >= READINESS_THRESHOLDS["competency_vs_target_grade"])},
        ]

        # Structural ceiling. A non-head cannot step past the grade of the head
        # of their own unit without the step being a different move (a transfer
        # or a restructure), so the label reports it rather than scoring a
        # promotion the org chart has no room for.
        head_row = self._head_of_unit.get(int(self.unit_of[row]))
        manager: dict[str, Any] = {"unit_id": int(self.unit_of[row])}
        blocked = False
        if head_row is not None and head_row != row:
            manager.update(person_id=self.person_id[head_row],
                           employee_id=self.employee_id[head_row],
                           grade_level=int(self.grade_level[head_row]))
            blocked = target > int(self.grade_level[head_row])
        elif head_row == row:
            manager.update(person_id=None, note="the person is the head of this unit")
        else:
            manager.update(person_id=None, note="no head recorded for this unit")
        criteria.append({"name": "grade_headroom_under_manager",
                         "value": manager.get("grade_level"),
                         "threshold": target, "passed": not blocked})

        passed = [c["passed"] for c in criteria[:3]]
        score = float(np.mean([perf, pot, self.competency_pct[row]]))
        if all(passed) and not blocked:
            verdict = "ready_now"
        elif passed[0] and passed[1]:
            verdict = "ready_with_development"
        elif passed[0] or passed[1]:
            verdict = "not_ready_yet"
        else:
            verdict = "not_ready"

        return Readiness(
            family="readiness_for_step",
            frame="snapshot",
            person_id=self.person_id[row],
            employee_id=self.employee_id[row],
            unknown_ref=None,
            current_grade=current,
            target_grade=target,
            verdict=verdict,
            readiness_score=_r(score),
            criteria=criteria,
            benchmark={"frame": benchmark_frame, "peers": int(peer_count),
                       "median_competency": _r(bench_comp),
                       "person_competency": _r(self.competency_avg[row])},
            manager=manager,
            method={
                "target_grade_rule": "current grade + 1 unless overridden",
                "thresholds": dict(READINESS_THRESHOLDS),
                "verdicts": ["ready_now", "ready_with_development",
                             "not_ready_yet", "not_ready", "unknown_person"],
                "note": ("blocked headroom demotes ready_now to "
                         "ready_with_development; it is a structural fact, not a "
                         "judgement about the person"),
            },
        )

    # ==================================================================
    # Family 4 — identifying key employees
    # ==================================================================

    def stale_key_employee_flag(self) -> np.ndarray:
        """Reproduce the corpus's ``is_key_employee`` column. **Not a label.**

        This is last year's methodology, kept in the corpus by
        ``b2e/traps.py::stale_key_employee`` precisely so that an agent which
        reads the ready-made flag instead of applying the current method lands
        somewhere measurably different from the reference answer.

        It is re-derived here from the population and the snapshot seed, exactly
        as ``b2e.gen.resolve._key_employee`` derives it, and never read out of a
        ``.col`` file — reading the column would break this module's one rule and
        would also make the reproduction untestable, since agreement with the
        column would then be a tautology.

        One caveat, and it is a real one rather than a formality. ``truth`` stores
        ``perf_latent`` and ``competency_avg`` rounded to four decimals, while the
        generator compared the unrounded values against its cut-offs. A person
        sitting within half a unit of the last decimal of either cut-off is
        therefore *undecidable* from the population as persisted, and this
        reproduction may put them on the wrong side.

        ``competency_avg`` is the case that actually bites: it is the mean of
        nine values on a 0.1 grid, so a sizeable group of people land on exactly
        3.60, which is exactly the cut. ``stale_flag_undecidable`` returns that
        set, ``key_employees`` reports its size, and the verification script
        asserts that every disagreement with the served column falls inside it.
        Reproducing the float32 arithmetic instead would mean rebuilding the
        competency matrix — that is, reading the corpus rather than the
        population — which this module does not do for any reason.
        """
        idx = np.arange(self.n)
        perf = self.perf_latent + 0.35 * normal(key64(self.seed, "key.lastyear"), idx)
        return ((perf > 0.75) & (self.competency_avg >= 3.6))

    def stale_flag_undecidable(self) -> np.ndarray:
        """Rows where the stale flag cannot be recovered from rounded truth.

        Half of the last stored decimal is the whole uncertainty: values were
        written with ``np.round(..., 4)``, so the generator's number was within
        5e-5 of what is on disk. A person that close to either cut-off could
        have been on either side of it.
        """
        tol = 0.5 * 10 ** -ROUND_DP
        idx = np.arange(self.n)
        perf = self.perf_latent + 0.35 * normal(key64(self.seed, "key.lastyear"), idx)
        return (np.abs(perf - 0.75) <= tol) | (np.abs(self.competency_avg - 3.6) <= tol)

    def key_employee_mask(self) -> np.ndarray:
        """The CORRECT methodology, applied snapshot-wide.

        Three routes in, and each answers a different question about why losing
        someone would hurt:

        * ``impact_pct >= KEY_IMPACT_PCT`` — sheer contribution;
        * a unit head at ``KEY_HEAD_IMPACT_PCT`` — structural criticality, since
          a head's departure costs the unit its continuity regardless of where
          they sit in the org-wide distribution;
        * a scarce senior grade at ``KEY_SCARCE_IMPACT_PCT`` — replacement cost,
          because there are only a few dozen people at G17+ and the market for
          them is thin.

        The frame is the whole snapshot, not the person's own unit. "Key" is an
        organisational statement: a unit-relative definition would make every
        team of ten contain exactly one key employee by construction, which is
        arithmetic dressed as insight.
        """
        return ((self.impact_pct >= KEY_IMPACT_PCT)
                | (self.is_head & (self.impact_pct >= KEY_HEAD_IMPACT_PCT))
                | ((self.grade_level >= KEY_SCARCE_GRADE)
                   & (self.impact_pct >= KEY_SCARCE_IMPACT_PCT)))

    def key_employees(self, unit_id: int | None = None,
                      recursive: bool = True, top_k: int | None = None) -> KeyEmployees:
        """Who is key in a scope, plus the measured disagreement with the flag.

        ``stale_flag_disagreement`` is reported on every call and is the reason
        this family is in the basket at all. Without it, an agent that answered
        ``SELECT ... WHERE is_key_employee = 1`` and an agent that applied the
        methodology would be indistinguishable on any snapshot where the two
        happened to overlap; with it, the gap is a number.
        """
        rows = self.members(unit_id, recursive)
        gold = self.key_employee_mask()
        stale = self.stale_key_employee_flag()
        undecidable = self.stale_flag_undecidable()

        key_rows = [int(r) for r in rows[gold[rows]]]
        key_rows.sort(key=lambda r: (-float(self.impact[r]), self.person_id[r]))
        if top_k is not None:
            key_rows = key_rows[:top_k]

        people: list[dict[str, Any]] = []
        for position, row in enumerate(key_rows, start=1):
            entry = self.card(row)
            entry["rank"] = position
            entry["reasons"] = [
                name for name, ok in (
                    ("top_impact", self.impact_pct[row] >= KEY_IMPACT_PCT),
                    ("unit_head", bool(self.is_head[row])
                     and self.impact_pct[row] >= KEY_HEAD_IMPACT_PCT),
                    ("scarce_grade", self.grade_level[row] >= KEY_SCARCE_GRADE
                     and self.impact_pct[row] >= KEY_SCARCE_IMPACT_PCT),
                ) if ok
            ]
            entry["stale_flag"] = bool(stale[row])
            people.append(entry)

        at_risk = [e["person_id"] for e in people if e["attrition_risk"] == "high"]

        scope_gold = gold[rows]
        scope_stale = stale[rows]
        both = int((scope_gold & scope_stale).sum())
        union = int((scope_gold | scope_stale).sum())
        return KeyEmployees(
            family="key_employees",
            frame="snapshot",
            scope={"unit_id": unit_id, "recursive": recursive,
                   "unit_name": (str(self.tree.name[unit_id])
                                 if unit_id is not None else None),
                   "unit_path": (self.unit_path(unit_id)
                                 if unit_id is not None else [])},
            population=int(rows.size),
            key_person_ids=[e["person_id"] for e in people],
            key_at_risk_person_ids=at_risk,
            people=people,
            stale_flag_disagreement={
                "gold_count": int(scope_gold.sum()),
                "stale_flag_count": int(scope_stale.sum()),
                "agree_both": both,
                "gold_only": int((scope_gold & ~scope_stale).sum()),
                "stale_only": int((~scope_gold & scope_stale).sum()),
                "jaccard": _r(both / union) if union else None,
                "undecidable_rows": int(undecidable[rows].sum()),
                "warning": ("is_key_employee is last year's snapshot and is NOT "
                            "the reference answer; it is reported only so the "
                            "disagreement can be measured"),
                "undecidable_note": ("rows sitting on a cut-off of the stale "
                                     "methodology, where the reproduction from "
                                     "rounded truth may differ from the served "
                                     "column; an upper bound on the noise in the "
                                     "counts above"),
            },
            method={
                "frame": "snapshot-wide impact percentile",
                "routes": {"top_impact": KEY_IMPACT_PCT,
                           "unit_head": KEY_HEAD_IMPACT_PCT,
                           "scarce_grade": [KEY_SCARCE_GRADE, KEY_SCARCE_IMPACT_PCT]},
                "weights": dict(IMPACT_WEIGHTS),
                "tie_break": "person_id ascending",
            },
        )

    # ==================================================================
    # Family 5 — team analysis
    # ==================================================================

    def team(self, unit_id: int, recursive: bool = True) -> TeamProfile:
        """A full structured profile of one unit.

        Everything comparative in here is a percentile of the snapshot or a rank
        among peer units, never a raw latent value. A team question is almost
        always relative — "is this team strong", "where is the risk" — and an
        absolute mean of a standardised factor answers none of them.

        ``peer_ranking`` ranks the unit among units at the same tree level with
        at least ``MIN_PEER_UNIT_SIZE`` members. The size floor is not cosmetic:
        without it a two-person unit tops the ranking whenever one of the two is
        strong, and the label would be teaching the agent noise.
        """
        unit_id = int(unit_id)
        rows = self.members(unit_id, recursive)
        direct = self.members(unit_id, recursive=False)
        gold_key = self.key_employee_mask()

        grades = self.grade_level[rows]
        perf = self.performance_pct[rows]
        comp = self.competency_avg[rows]
        risk = self.attrition_risk[rows]

        head_row = self._head_of_unit.get(unit_id)
        if head_row is None and recursive:
            # Upper units hold no people of their own; their head is the head of
            # the strongest-graded sub-unit, which is how population._appoint_heads
            # builds the chain. Reproducing the choice here keeps the answer to
            # "who runs this department" defined at every level of the tree.
            candidates = [r for r in rows if self.is_head[r]]
            if candidates:
                head_row = max(candidates, key=lambda r: (int(self.grade_level[r]),
                                                          -int(r)))

        high_risk_rows = [int(r) for r in rows[risk == 2]]
        high_risk_rows.sort(key=lambda r: (-float(self.impact[r]), self.person_id[r]))

        key_rows = [int(r) for r in rows[gold_key[rows]]]
        key_rows.sort(key=lambda r: (-float(self.impact[r]), self.person_id[r]))

        ready = [self.readiness(self.person_id[int(r)]) for r in rows]
        ready_now = [r.person_id for r in ready if r.verdict == "ready_now"]

        level = int(self.tree.level[unit_id])
        rank, peer_count, peer_value = self._peer_rank(unit_id, level)

        sub_units: list[dict[str, Any]] = []
        for child in sorted(self._children.get(unit_id, ())):
            child_rows = self.members(child, recursive=True)
            sub_units.append({
                "unit_id": child,
                "unit_name": str(self.tree.name[child]),
                "kind": str(self.tree.kind[child]),
                "headcount": int(child_rows.size),
                "mean_performance_pct": (_r(self.performance_pct[child_rows].mean())
                                         if child_rows.size else None),
                "high_attrition_risk": int((self.attrition_risk[child_rows] == 2).sum()),
            })

        return TeamProfile(
            family="team_analysis",
            frame="snapshot",
            unit={"unit_id": unit_id, "name": str(self.tree.name[unit_id]),
                  "kind": str(self.tree.kind[unit_id]),
                  "level": level, "path": self.unit_path(unit_id),
                  "recursive": recursive},
            headcount={"in_scope": int(rows.size), "direct_members": int(direct.size),
                       "sub_units": len(self._children.get(unit_id, ())),
                       "span_of_control": int(direct.size) - (1 if head_row in direct.tolist() else 0)},
            head=(self.card(head_row) if head_row is not None
                  else {"person_id": None, "note": "no head recorded in this scope"}),
            grades={"min": int(grades.min()) if rows.size else None,
                    "max": int(grades.max()) if rows.size else None,
                    "median": _r(np.median(grades)) if rows.size else None,
                    "distribution": {str(g): int(c) for g, c in
                                     zip(*np.unique(grades, return_counts=True))}},
            performance={"mean_pct": _r(perf.mean()) if rows.size else None,
                         "median_pct": _r(np.median(perf)) if rows.size else None,
                         "quartiles_pct": _pct_list(perf, (0.25, 0.5, 0.75)),
                         "above_snapshot_median": int((perf >= 0.5).sum()),
                         "top_decile_members": int((perf >= 0.9).sum())},
            competency={"mean": _r(comp.mean()) if rows.size else None,
                        "median": _r(np.median(comp)) if rows.size else None,
                        "min": _r(comp.min()) if rows.size else None,
                        "max": _r(comp.max()) if rows.size else None},
            attrition={"low": int((risk == 0).sum()), "medium": int((risk == 1).sum()),
                       "high": int((risk == 2).sum()),
                       "high_risk_person_ids": [self.person_id[r] for r in high_risk_rows],
                       "high_risk_share": _r(float((risk == 2).mean())) if rows.size else None},
            key_people={"count": len(key_rows),
                        "person_ids": [self.person_id[r] for r in key_rows],
                        "at_risk_person_ids": [self.person_id[r] for r in key_rows
                                               if self.attrition_risk[r] == 2],
                        "share": _r(len(key_rows) / rows.size) if rows.size else None},
            succession={"ready_now_count": len(ready_now),
                        "ready_now_person_ids": ready_now,
                        "ready_with_development_count":
                            sum(1 for r in ready if r.verdict == "ready_with_development")},
            peer_ranking={"level": level,
                          "min_unit_size": MIN_PEER_UNIT_SIZE,
                          "metric": "mean performance percentile over the subtree",
                          "rank": rank, "peer_units": peer_count,
                          "value": peer_value},
            sub_units=sub_units,
            method={
                "frame": "percentiles are snapshot-wide",
                "peer_frame": f"units at tree level {level} with >= "
                              f"{MIN_PEER_UNIT_SIZE} members",
                "key_employee_method": "GoldLabels.key_employee_mask",
                "note": ("truth carries no employment status, so terminated "
                         "employees are inside the headcount; word team "
                         "questions as 'in the snapshot', not 'active'"),
            },
        )
