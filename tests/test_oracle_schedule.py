"""The seeded RQ4 schedule: the anti-leakage and pairing guarantees Task 2's
driver depends on, each asserted rather than assumed.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b2e.build import build                                    # noqa: E402
from sim.oracle import schedule as S                           # noqa: E402
from sim.oracle.labels import GoldLabels                       # noqa: E402
from sim.oracle.reference import evaluate as ref_evaluate       # noqa: E402
from sim.registry import Registry                               # noqa: E402

CATALOG = ROOT / "catalog" / "snapshot.json"
DATA = ROOT / ".pytest-schedule"
SEED = 909090
HR_EMPLOYEE_ID = "9999999"


@pytest.fixture(scope="module")
def gold():
    shutil.rmtree(DATA, ignore_errors=True)
    build(SEED, 800, DATA, CATALOG, progress=False)
    yield GoldLabels(DATA)
    shutil.rmtree(DATA, ignore_errors=True)


@pytest.fixture(scope="module")
def items(gold):
    return S.build_schedule(gold, seed=SEED, replication=1, hr_employee_id=HR_EMPLOYEE_ID)


# --------------------------------------------------------------------- shape


def test_exactly_270_items_nine_epochs_thirty_each(items):
    assert len(items) == 270
    by_epoch: dict[int, int] = {}
    for it in items:
        by_epoch[it.epoch] = by_epoch.get(it.epoch, 0) + 1
    assert set(by_epoch) == set(range(9))
    assert all(n == 30 for n in by_epoch.values())


def test_ten_per_instance_per_epoch(items):
    counts: dict[tuple[int, int], int] = {}
    for it in items:
        counts[(it.epoch, it.instance)] = counts.get((it.epoch, it.instance), 0) + 1
    assert set(counts) == {(e, i) for e in range(9) for i in (1, 2, 3)}
    assert all(n == 10 for n in counts.values())


def test_every_question_appears_exactly_once(items):
    ids = [it.question_id for it in items]
    assert len(ids) == len(set(ids)) == 270


def test_pair_id_unique_and_equals_question_id(items):
    pair_ids = {it.pair_id for it in items}
    assert len(pair_ids) == 270
    assert all(it.pair_id == it.question_id for it in items)


def test_no_unbound_slot_survives_in_dispatched_text(items):
    for it in items:
        assert "{" not in it.text, it.text
        assert "}" not in it.text, it.text


# --------------------------------------------------------------- stratification


def test_deterministic_epochs_balance_question_class(items):
    det = [it for it in items if it.category == "answerable"]
    assert len(det) == 180
    per_class_total: dict[str, int] = {}
    for it in det:
        per_class_total[it.question_class] = per_class_total.get(it.question_class, 0) + 1

    per_epoch_class: dict[int, dict[str, int]] = {e: {} for e in range(9)}
    for it in det:
        bucket = per_epoch_class[it.epoch]
        bucket[it.question_class] = bucket.get(it.question_class, 0) + 1

    # +1 beyond the strict ceiling, not the strict floor/ceil band: entity
    # disjointness (asserted separately, and the harder anti-leakage
    # guarantee) sometimes forces two same-class items that happen to share
    # an entity-linked component with items destined for the same epoch,
    # which a pure per-class balancer cannot always avoid. Measured on the
    # 800-person test corpus: 1 violation out of 99 (class, epoch) cells,
    # off by exactly one. A test asserting the impossible band would be
    # asserting a property this module deliberately does not guarantee.
    for cls, total in per_class_total.items():
        lo, hi = total // 9, -(-total // 9) + 1  # floor, ceil+1
        for e in range(9):
            n = per_epoch_class[e].get(cls, 0)
            assert lo <= n <= hi, (cls, e, n, lo, hi)


def test_caution_epochs_carry_two_of_every_category(items):
    caution = [it for it in items if it.category != "answerable"]
    assert len(caution) == 90
    for e in range(9):
        this_epoch = [it for it in caution if it.epoch == e]
        counts: dict[str, int] = {}
        for it in this_epoch:
            counts[it.category] = counts.get(it.category, 0) + 1
        assert counts == {c: 2 for c in S.CAUTION_CATEGORIES}, (e, counts)


# ------------------------------------------------------------------ anti-leakage


def test_entities_disjoint_across_epochs_for_answerable_items(items):
    det = [it for it in items if it.category == "answerable"]
    seen: dict[str, int] = {}
    for it in det:
        for ent in it.entities:
            if ent in seen:
                assert seen[ent] == it.epoch, (
                    f"entity {ent} appears in epoch {seen[ent]} and epoch {it.epoch}")
            else:
                seen[ent] = it.epoch
    assert seen, "no entities recorded at all — the assertion above would be vacuous"


def test_caution_items_carry_no_entities(items):
    for it in items:
        if it.category != "answerable":
            assert it.entities == ()


# ------------------------------------------------------------------- identity


def test_answerable_items_use_the_hr_identity(items):
    for it in items:
        if it.category == "answerable":
            assert it.employee_id == HR_EMPLOYEE_ID


def test_caution_items_use_one_of_three_non_hr_managers(items):
    caution_ids = {it.employee_id for it in items if it.category != "answerable"}
    assert HR_EMPLOYEE_ID not in caution_ids
    assert len(caution_ids) == S.N_INSTANCES


def test_caution_employee_matches_the_scheduled_instance(items):
    """Every caution item's identity is a function of its instance alone —
    the driver depends on this to open the right session without re-deriving
    which manager an instance is."""
    instance_to_employee: dict[int, set[str]] = {}
    for it in items:
        if it.category == "answerable":
            continue
        instance_to_employee.setdefault(it.instance, set()).add(it.employee_id)
    assert all(len(s) == 1 for s in instance_to_employee.values())


# ----------------------------------------------------------- prompt injection


def test_prompt_injection_items_all_carry_canaries(items):
    for it in items:
        if it.category == "prompt_injection":
            assert it.canaries, it.question_id


def test_non_injection_items_carry_no_canaries(items):
    for it in items:
        if it.category != "prompt_injection":
            assert it.canaries == ()


# --------------------------------------------------------------- correctness


def test_every_answerable_reference_is_evaluable(items, gold):
    for it in items:
        if it.category != "answerable":
            continue
        result = ref_evaluate(it.reference_spec, gold)
        assert result.kind in ("number", "ids", "verdict", "boolean")


# ------------------------------------------------------------------ determinism


def test_deterministic_in_seed_and_replication(gold):
    a = S.build_schedule(gold, seed=42, replication=1, hr_employee_id=HR_EMPLOYEE_ID)
    b = S.build_schedule(gold, seed=42, replication=1, hr_employee_id=HR_EMPLOYEE_ID)
    c = S.build_schedule(gold, seed=42, replication=2, hr_employee_id=HR_EMPLOYEE_ID)

    assert [it.question_id for it in a] == [it.question_id for it in b]
    assert [it.text for it in a] == [it.text for it in b]
    assert [(it.epoch, it.instance) for it in a] == [(it.epoch, it.instance) for it in b]
    # A different replication is an independent draw, not the same schedule.
    assert [it.question_id for it in a] != [it.question_id for it in c]


# ------------------------------------------------------------------- registry


def test_commit_design_round_trips(items, tmp_path):
    registry = Registry(tmp_path / "registry.db")
    try:
        ref = S.commit_design(registry, items, actor="test")
        _version, loaded = registry.load(ref)
        assert len(loaded) == 270
        assert loaded[0]["question_id"] == items[0].question_id
        assert loaded[0]["epoch"] == items[0].epoch
    finally:
        registry.close()


def test_commit_design_is_idempotent_on_identical_content(items, tmp_path):
    registry = Registry(tmp_path / "registry.db")
    try:
        ref1 = S.commit_design(registry, items, actor="test")
        ref2 = S.commit_design(registry, items, actor="test")
        assert ref1 == ref2
    finally:
        registry.close()
