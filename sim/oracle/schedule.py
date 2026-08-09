"""The seeded RQ4 schedule: 270 questions, drawn once, replayed by every arm.

Why this exists
----------------
RQ4 compares A1 (no memory) against A2 (per-instance reflection) and A3 (pooled
reflection). The comparison is only paired if all three arms answer *the same*
270 questions, bound to *the same* entities, in the *same* epoch — otherwise a
gap between arms could be a gap between question sets rather than a gap between
memory strategies. This module draws that one schedule per replication and
hands it to the driver (``scripts/run_rq4.py``), which replays it unchanged
for A1, A2 and A3.

A defect this module works around, not one it was asked to solve
------------------------------------------------------------------
The plan that specified this module gave ``ScheduledItem`` five identifying
fields and no acting identity. Probing the live deployment
(``sim/emulator/identity.py``) found why that gap is not survivable:

* Heimdall enforces **row-level** scope for every non-``hr`` identity —
  ``restrict()`` silently ANDs a "rows I may see" filter onto every query, even
  one with no person filter at all. A ``manager`` identity sees only their own
  org subtree.
* ``sim.oracle.qgen``'s 180 deterministic questions pick units and people from
  the *whole* organisation, not from any one manager's subtree. Asked by a
  ``manager`` or ``self`` identity, a question like "how many grade>=14 in unit
  X" would silently come back scoped to the asker's own subtree — wrong for
  every unit outside it, not merely hard. Only the deployment's one ``hr``
  identity (``HR_EMPLOYEE_IDS`` in ``deploy/.env``) can answer them correctly.
* The basket's ``access_control`` caution questions need the *opposite*:
  ``sim.oracle.basket._check_gold_ref`` already refuses ``acting_role="hr"``
  outright, because an identity that sees everything can never produce the 403
  those questions are built around.

One fixed identity per instance cannot satisfy both. So this module binds each
item to the identity its category actually needs — the shared HR identity for
every ``answerable`` (qgen) item, and one of three ordinary, per-instance
manager identities for the 90 caution items — and stamps the choice onto
``ScheduledItem.employee_id``. Every session for a question still opens fresh
(``conversation_mode="stateless"``), so employee identity varying by question
within one instance's epoch slate is not a new constraint; it is what the
harness already does per turn. This is the reason ``ScheduledItem`` below
carries two fields ("employee_id", "entities") the plan's field list did not
list — see ``docs/superpowers/plans/2026-08-09-rq4-driver-and-analysis.md``
Task 1 and the accompanying report for the full account.

Why entity disjointness is checked only on ``answerable`` items
-----------------------------------------------------------------
The anti-leakage property ("no ``person:``/``unit:`` entity recurs in two
epochs") exists so a memorised *fact* about an entity in epoch *e* cannot help
answer a later factual question about the same entity in epoch *e'* — that
risk exists only for qgen's answerable items, whose correct answer is exactly
such a fact. A caution item's correct answer is a *behaviour* (decline, ask,
report no data, ignore an injection, get denied) that does not depend on which
person or unit was named, so caution items reuse their instance's own manager
identity (and that manager's own unit) across every epoch by construction —
there is nothing to leak. ``ScheduledItem.entities`` is populated for
``category == "answerable"`` items only; caution items carry an empty tuple.

How the 270 items land on the 9x3 grid
-----------------------------------------
Two independent placement passes, because the two halves have different
constraints:

* The 180 deterministic items are grouped into entity-sharing components
  (union-find over ``entities``, since two items naming the same person/unit
  must land in the same epoch), then components are assigned to epochs by a
  greedy rule that always extends the currently-least-loaded epoch *for that
  item's question_class* — this is what keeps each epoch's class mix even
  without ever needing to search for an optimum. Instance assignment within an
  epoch happens afterwards, once the caution half (below) has fixed how many
  caution slots each instance already holds that epoch: each instance gets
  ``10 - caution_count`` deterministic items, dealt out by a seeded shuffle.
* The 90 caution items need no entity bookkeeping (see above) but do need a
  fixed instance *before* binding, because binding reads that instance's
  manager scope. For category index ``ci`` and item index ``k`` in
  ``0..17``: ``epoch = k % 9``, ``instance = (k % 9 + k // 9 + ci) % 3``. Over
  one category's 18 items this visits every ``(epoch, instance)`` pair the
  right number of times: each epoch gets exactly 2 items (``18 / 9``), the two
  always land on *different* instances (the ``k // 9`` term flips the
  instance between an item's first and second pass through the epoch range),
  and each instance ends up with exactly 6 of the 18 (``18 / 3``). Offsetting
  by ``ci`` keeps five categories from all favouring the same instance in the
  same epoch.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from sim.oracle import qgen
from sim.oracle.basket import Question, injection_canaries, load_basket
from sim.oracle.labels import GoldLabels
from sim.oracle.reference import ReferenceSpec
from sim.registry import Registry

N_EPOCHS = 9
N_INSTANCES = 3
QUESTIONS_PER_EPOCH = 30
QUESTIONS_PER_INSTANCE_PER_EPOCH = QUESTIONS_PER_EPOCH // N_INSTANCES  # 10
DET_TOTAL = 180
DET_PER_EPOCH = DET_TOTAL // N_EPOCHS  # 20

#: The five non-answerable categories drawn for caution coverage, and how many
#: of each — 18 per category (6 per instance) x 5 = 90, matching Task 1.
CAUTION_CATEGORIES: tuple[str, ...] = (
    "out_of_scope", "ambiguous", "no_data", "access_control", "prompt_injection",
)
CAUTION_PER_CATEGORY = 18
CAUTION_TOTAL = CAUTION_PER_CATEGORY * len(CAUTION_CATEGORIES)  # 90
CAUTION_PER_EPOCH = CAUTION_TOTAL // N_EPOCHS  # 10
CAUTION_PER_CATEGORY_PER_EPOCH = CAUTION_PER_CATEGORY // N_EPOCHS  # 2

#: A unit's subtree must have at least this many people before an instance's
#: manager is drawn from it — small enough that "my colleague" / "my team"
#: caution wording still reads as true, large enough that an in-scope and an
#: out-of-scope person are both easy to find without collisions.
_MIN_MANAGER_SUBTREE = 8

_ANSWERABLE_CATEGORY = "answerable"


@dataclass(frozen=True, slots=True)
class ScheduledItem:
    """One question, bound and placed on the calendar.

    ``question_id`` and ``pair_id`` are the same string. They are two fields
    rather than one because they mean two different things to two different
    readers: ``question_id`` is this item's identity for the checkpoint and
    ``turns.jsonl``; ``pair_id`` is the key Task 3's paired test groups on.
    Because the schedule is arm-agnostic (the same 270 items are replayed for
    A1, A2 and A3 — see the module docstring), the two happen to coincide, but
    a caller should read the field whose *meaning* it needs rather than assume
    the coincidence is permanent.
    """

    question_id: str
    text: str
    category: str
    family: str
    question_class: str
    reference_spec: ReferenceSpec | None
    canaries: tuple[str, ...]
    epoch: int
    instance: int
    pair_id: str
    #: Acting identity for this question's session. See the module docstring
    #: for why this is not a per-instance constant.
    employee_id: str
    #: ``person:<id>`` / ``unit:<id>`` entities this item's correct answer
    #: depends on. Populated for ``category == "answerable"`` only.
    entities: tuple[str, ...]


# --------------------------------------------------------------------- hashing


def _h(seed: int, *parts: Any) -> int:
    """A stable integer from a seed and any tuple of parts. Mirrors
    ``sim.oracle.qgen._h`` so the two modules hash the same way, but is
    reimplemented locally rather than imported — it is a five-line pure
    function and importing a leading-underscore name across modules would
    read as a real dependency where none is intended."""
    raw = "|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _pick(items: list[Any], seed: int, *parts: Any) -> Any:
    return items[_h(seed, *parts) % len(items)]


def _effective_seed(seed: int, replication: int) -> int:
    """Folds ``replication`` into ``seed`` once, so every draw below only has
    to thread one integer instead of two, and so two replications of the same
    ``seed`` are independent draws rather than the same schedule twice."""
    return _h(seed, "rq4_schedule", replication)


# -------------------------------------------------------------- instance picks


@dataclass(frozen=True, slots=True)
class _Manager:
    instance: int
    row: int
    person_id: str
    unit_id: int
    #: Row indices of this manager's own subtree (themselves included).
    scope_rows: tuple[int, ...]


def _pick_instance_managers(gold: GoldLabels, eff_seed: int) -> tuple[_Manager, ...]:
    """Three distinct, deterministically-chosen heads with a large enough
    subtree to bind caution questions against. Raises loudly (matching
    ``qgen``'s own convention) rather than silently handing back fewer than
    three — a schedule with fewer real instances would silently shrink the
    experiment's design.
    """
    eligible = [
        i for i in range(gold.n)
        if bool(gold.is_head[i])
        and len(gold.members(int(gold.unit_of[i]), True)) >= _MIN_MANAGER_SUBTREE
    ]
    if len(eligible) < N_INSTANCES:
        raise ValueError(
            f"only {len(eligible)} head(s) with a subtree >= {_MIN_MANAGER_SUBTREE} "
            f"in this snapshot; need {N_INSTANCES} distinct instance managers")

    # Deterministic order, then take the first three distinct picks — sorting
    # by hash rather than by row index so the choice depends on the seed, not
    # on snapshot build order.
    ordered = sorted(eligible, key=lambda i: _h(eff_seed, "instance_pick", i))
    chosen = ordered[:N_INSTANCES]

    out = []
    for k, row in enumerate(chosen, start=1):
        unit = int(gold.unit_of[row])
        scope_rows = tuple(int(r) for r in gold.members(unit, True))
        out.append(_Manager(instance=k, row=row, person_id=gold.person_id[row],
                            unit_id=unit, scope_rows=scope_rows))
    return tuple(out)


def _in_scope_person(manager: _Manager, gold: GoldLabels, eff_seed: int, *parts: Any,
                     avoid: str | None = None) -> str:
    """A person in ``manager``'s own subtree — plausible as "my colleague"."""
    candidates = [r for r in manager.scope_rows if gold.person_id[r] != avoid] \
        or list(manager.scope_rows)
    row = _pick(candidates, eff_seed, "in_scope", manager.instance, *parts)
    return gold.person_id[row]


def _out_of_scope_person(manager: _Manager, gold: GoldLabels, eff_seed: int,
                         *parts: Any) -> str:
    """A person outside ``manager``'s subtree — genuinely denies under
    Heimdall row-level scope, so an ``access_control`` (``denied:row``)
    caution question actually exercises the 403 it is built around."""
    scope = set(manager.scope_rows)
    outside = [i for i in range(gold.n) if i not in scope]
    row = _pick(outside, eff_seed, "out_of_scope", manager.instance, *parts)
    return gold.person_id[row]


def _requisition(eff_seed: int, *parts: Any) -> str:
    """A syntactically plausible vacancy id. No caution question's correctness
    depends on this string naming a real requisition — the categories here
    score a behaviour (decline/ask/report/ignore/deny), never a value — so a
    deterministic synthetic id is exactly as good as a real one and needs no
    dependency on the HR-only recruitment schema."""
    return f"REQ-2026-{100000 + _h(eff_seed, 'requisition', *parts) % 9000}"


# ------------------------------------------------------------ deterministic 180


@dataclass(frozen=True, slots=True)
class _Placed:
    """A candidate item with its placement already decided."""

    question_id: str
    text: str
    category: str
    family: str
    question_class: str
    reference_spec: ReferenceSpec | None
    canaries: tuple[str, ...]
    employee_id: str
    entities: tuple[str, ...]
    epoch: int
    instance: int


def _det_epoch_assignment(gold: GoldLabels, eff_seed: int,
                          hr_employee_id: str) -> list[_Placed]:
    """Places the 180 qgen items into epochs (instance left as 0, filled in by
    ``_assign_det_instances``), respecting entity disjointness and per-class
    balance. See the module docstring's "How the 270 items land" section.
    """
    generated = list(qgen.generate(gold, seed=eff_seed))
    if len(generated) != DET_TOTAL:
        raise RuntimeError(
            f"qgen.generate produced {len(generated)}, expected {DET_TOTAL}")

    # Union-find over shared entities: two items naming the same person/unit
    # must land in the same epoch, or a memorised fact about that entity from
    # one epoch could leak into a later one.
    parent = {gq.id: gq.id for gq in generated}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_entity: dict[str, list[str]] = {}
    for gq in generated:
        for ent in gq.entities:
            by_entity.setdefault(ent, []).append(gq.id)
    for ids in by_entity.values():
        for other in ids[1:]:
            union(ids[0], other)

    components: dict[str, list[str]] = {}
    for gq in generated:
        components.setdefault(find(gq.id), []).append(gq.id)

    by_id = {gq.id: gq for gq in generated}
    # Largest components first ("best-fit-decreasing"): with zero slack
    # (180 items exactly fill 9 x 20 epochs), placing a size-2+ component
    # while every epoch still has room is what makes the packing feasible at
    # all — deferring it to the end, after singletons have already narrowed
    # every epoch's remaining capacity, is exactly how a impossible-to-place
    # remainder happens. Size ties (nearly everything, since most components
    # are singletons) break by hash, so ordering among same-size components
    # is still a pure function of the seed.
    ordered_components = sorted(
        components.values(),
        key=lambda ids: (-len(ids), _h(eff_seed, "component_order", tuple(sorted(ids)))))

    epoch_total = [0] * N_EPOCHS
    epoch_class_count: list[dict[str, int]] = [dict() for _ in range(N_EPOCHS)]

    def cost(epoch: int, ids: list[str]) -> tuple[int, int]:
        if epoch_total[epoch] + len(ids) > DET_PER_EPOCH:
            return (10**9, 10**9)
        class_load = sum(
            epoch_class_count[epoch].get(by_id[i].question_class, 0) for i in ids)
        return (class_load, epoch_total[epoch])

    placement: dict[str, int] = {}
    for ids in ordered_components:
        best_epoch = min(range(N_EPOCHS), key=lambda e: (cost(e, ids), e))
        if cost(best_epoch, ids)[0] >= 10**9:
            raise RuntimeError(
                "could not place every deterministic item within capacity; "
                "an entity-sharing component is larger than an epoch")
        for i in ids:
            placement[i] = best_epoch
            epoch_total[best_epoch] += 1
            cls = by_id[i].question_class
            epoch_class_count[best_epoch][cls] = \
                epoch_class_count[best_epoch].get(cls, 0) + 1

    return [
        _Placed(question_id=gq.id, text=gq.text, category=_ANSWERABLE_CATEGORY,
               family=gq.family, question_class=gq.question_class,
               reference_spec=gq.spec, canaries=(), employee_id=hr_employee_id,
               entities=gq.entities, epoch=placement[gq.id], instance=0)
        for gq in generated
    ]


def _assign_det_instances(det: list[_Placed], caution_epoch_instance_count:
                          dict[int, dict[int, int]], eff_seed: int) -> list[_Placed]:
    """Fills in ``instance`` on the deterministic items, epoch by epoch, so
    that every ``(epoch, instance)`` cell ends at exactly
    ``QUESTIONS_PER_INSTANCE_PER_EPOCH`` once the caution items (already
    placed) are added in.
    """
    by_epoch: dict[int, list[_Placed]] = {}
    for item in det:
        by_epoch.setdefault(item.epoch, []).append(item)

    out: list[_Placed] = []
    for epoch, items in by_epoch.items():
        caution_counts = caution_epoch_instance_count.get(epoch, {})
        targets = {
            inst: QUESTIONS_PER_INSTANCE_PER_EPOCH - caution_counts.get(inst, 0)
            for inst in range(1, N_INSTANCES + 1)
        }
        if sum(targets.values()) != len(items):
            raise RuntimeError(
                f"epoch {epoch}: deterministic slots {sum(targets.values())} "
                f"!= items {len(items)}; caution placement did not leave "
                f"exactly {DET_PER_EPOCH} deterministic slots")
        # A label list with the right multiplicities, seed-shuffled, then
        # zipped against a deterministically-ordered item list.
        labels: list[int] = []
        for inst in range(1, N_INSTANCES + 1):
            labels.extend([inst] * targets[inst])
        ordered_labels = sorted(
            range(len(labels)), key=lambda i: _h(eff_seed, "det_instance_shuffle",
                                                  epoch, i))
        shuffled = [labels[i] for i in ordered_labels]
        ordered_items = sorted(
            items, key=lambda it: _h(eff_seed, "det_item_order", epoch, it.question_id))
        for item, inst in zip(ordered_items, shuffled):
            out.append(_Placed(
                question_id=item.question_id, text=item.text, category=item.category,
                family=item.family, question_class=item.question_class,
                reference_spec=item.reference_spec, canaries=item.canaries,
                employee_id=item.employee_id, entities=item.entities,
                epoch=item.epoch, instance=inst))
    return out


# ----------------------------------------------------------------- caution 90


def _caution_pool(category: str) -> list[Question]:
    pool = list(load_basket(categories=[category]))
    if category == "prompt_injection":
        # Defensive, not reachable on today's basket (every prompt_injection
        # seed maps to a non-empty canary set) — see the module docstring's
        # forward-reference and sim.research.evaluate.score_correctness, which
        # raises outright on an empty canary tuple rather than silently
        # scoring every answer in the category correct.
        pool = [q for q in pool if injection_canaries(q)]
    return pool


def _bind_caution(q: Question, manager: _Manager, gold: GoldLabels,
                  eff_seed: int, idx: int) -> str:
    """Render one caution question's text against ``manager``'s scope."""
    values: dict[str, Any] = {}
    row_needs_denial = (q.category == "access_control"
                        and (q.gold_ref or "") == "denied:row")

    if "subject" in q.slots:
        values["subject"] = (
            _out_of_scope_person(manager, gold, eff_seed, q.id, idx, "subject")
            if row_needs_denial else
            _in_scope_person(manager, gold, eff_seed, q.id, idx, "subject"))
    if "peer" in q.slots:
        values["peer"] = (
            _out_of_scope_person(manager, gold, eff_seed, q.id, idx, "peer")
            if row_needs_denial else
            _in_scope_person(manager, gold, eff_seed, q.id, idx, "peer",
                             avoid=values.get("subject")))
    if "unit" in q.slots:
        values["unit"] = str(gold.tree.name[manager.unit_id])
    if "requisition" in q.slots:
        values["requisition"] = _requisition(eff_seed, q.id, idx)

    return q.bind(values)


def _caution_placement(gold: GoldLabels, eff_seed: int,
                       managers: tuple[_Manager, ...]) -> list[_Placed]:
    """Binds and places the 90 caution items. See the module docstring's
    "How the 270 items land" section for the ``(epoch, instance)`` formula.
    """
    out: list[_Placed] = []
    for ci, category in enumerate(CAUTION_CATEGORIES):
        pool = _caution_pool(category)
        if len(pool) < CAUTION_PER_CATEGORY:
            raise ValueError(
                f"basket category {category!r} has only {len(pool)} usable "
                f"question(s) after filtering, need {CAUTION_PER_CATEGORY}")
        # Deterministic order, independent of basket authoring order.
        ordered = sorted(pool, key=lambda q: _h(eff_seed, "caution_pick", category, q.id))
        chosen = ordered[:CAUTION_PER_CATEGORY]

        for k, q in enumerate(chosen):
            epoch = k % N_EPOCHS
            instance = (k % N_EPOCHS + k // N_EPOCHS + ci) % N_INSTANCES + 1
            manager = managers[instance - 1]
            text = _bind_caution(q, manager, gold, eff_seed, k)
            out.append(_Placed(
                question_id=f"{q.id}#r{eff_seed}", text=text, category=q.category,
                family=q.family, question_class=q.category, reference_spec=None,
                canaries=injection_canaries(q), employee_id=manager.person_id,
                entities=(), epoch=epoch, instance=instance))
    return out


def _caution_epoch_instance_counts(caution: list[_Placed]) -> dict[int, dict[int, int]]:
    counts: dict[int, dict[int, int]] = {}
    for item in caution:
        bucket = counts.setdefault(item.epoch, {})
        bucket[item.instance] = bucket.get(item.instance, 0) + 1
    return counts


# ------------------------------------------------------------------- assembly


def build_schedule(gold: GoldLabels, *, seed: int, replication: int,
                   hr_employee_id: str) -> tuple[ScheduledItem, ...]:
    """The 270-item schedule for one replication: 180 deterministic questions
    from :func:`sim.oracle.qgen.generate` plus 90 caution questions from
    :func:`sim.oracle.basket.load_basket`, placed across 9 epochs x 3
    instances x 10 questions.

    ``hr_employee_id`` is not in the plan's original signature — see the
    module docstring's "A defect this module works around" section for why it
    is required rather than optional: without it, every deterministic
    (``answerable``) question would be bound to an acting identity that
    Heimdall's row-level scope makes structurally unable to answer it
    correctly.
    """
    eff_seed = _effective_seed(seed, replication)
    managers = _pick_instance_managers(gold, eff_seed)

    caution = _caution_placement(gold, eff_seed, managers)
    caution_counts = _caution_epoch_instance_counts(caution)

    det_by_epoch = _det_epoch_assignment(gold, eff_seed, hr_employee_id)
    det = _assign_det_instances(det_by_epoch, caution_counts, eff_seed)

    placed = det + caution
    if len(placed) != DET_TOTAL + CAUTION_TOTAL:
        raise RuntimeError(
            f"assembled {len(placed)} items, expected {DET_TOTAL + CAUTION_TOTAL}")

    return tuple(
        ScheduledItem(
            question_id=p.question_id, text=p.text, category=p.category,
            family=p.family, question_class=p.question_class,
            reference_spec=p.reference_spec, canaries=p.canaries,
            epoch=p.epoch, instance=p.instance, pair_id=p.question_id,
            employee_id=p.employee_id, entities=p.entities,
        )
        for p in sorted(placed, key=lambda p: (p.epoch, p.instance, p.question_id))
    )


# --------------------------------------------------------------- registry I/O


def _spec_to_dict(spec: ReferenceSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {"op": spec.op, "field": spec.field, "scope": dict(spec.scope),
           "predicate": dict(spec.predicate) if spec.predicate else None,
           "n": spec.n, "refs": list(spec.refs)}


def _item_to_dict(item: ScheduledItem) -> dict[str, Any]:
    return {
        "question_id": item.question_id, "text": item.text, "category": item.category,
        "family": item.family, "question_class": item.question_class,
        "reference_spec": _spec_to_dict(item.reference_spec),
        "canaries": list(item.canaries), "epoch": item.epoch, "instance": item.instance,
        "pair_id": item.pair_id, "employee_id": item.employee_id,
        "entities": list(item.entities),
    }


def commit_design(registry: Registry, items: "tuple[ScheduledItem, ...]", *,
                  actor: str) -> str:
    """Append the schedule to the registry as a ``run_design`` artefact.

    Makes a run reproducible and re-scorable without re-running the agent: the
    exact 270 bound questions, their references and their placement are all
    recoverable from the ref this returns, the same way a committed basket or
    system prompt is.
    """
    version = registry.commit(
        "run_design", "run_design", [_item_to_dict(i) for i in items],
        actor=actor, note=f"{len(items)} items, {N_EPOCHS} epochs")
    return version.ref
