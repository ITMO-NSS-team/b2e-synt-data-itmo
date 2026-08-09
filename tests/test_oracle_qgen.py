"""The generated half of the basket.

Two properties matter more than the text: every question's reference must
evaluate without raising, and generation must be a pure function of the seed.
A generator that occasionally emits an unanswerable question would put a
permanent floor under every arm's score, and one that is not reproducible
would make a run impossible to re-score later.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b2e.build import build                                    # noqa: E402
from sim.oracle.labels import GoldLabels                       # noqa: E402
from sim.oracle import qgen, reference as ref                  # noqa: E402

CATALOG = ROOT / "catalog" / "snapshot.json"
DATA = ROOT / ".pytest-qgen"
SEED = 424242


@pytest.fixture(scope="module")
def gold():
    shutil.rmtree(DATA, ignore_errors=True)
    build(SEED, 800, DATA, CATALOG, progress=False)
    yield GoldLabels(DATA)
    shutil.rmtree(DATA, ignore_errors=True)


def test_generates_exactly_the_declared_class_counts(gold):
    questions = qgen.generate(gold, seed=1)

    assert len(questions) == 180
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.question_class] = counts.get(q.question_class, 0) + 1
    assert counts == qgen.CLASS_COUNTS


def test_every_generated_question_has_an_evaluable_reference(gold):
    for q in qgen.generate(gold, seed=1):
        result = ref.evaluate(q.spec, gold)
        assert result.kind in ("number", "ids", "verdict", "boolean")
        if result.kind == "number":
            assert result.value is not None, q.id
        if result.kind == "ids":
            assert result.ids, q.id


def test_generation_is_deterministic_in_the_seed(gold):
    a = qgen.generate(gold, seed=7)
    b = qgen.generate(gold, seed=7)
    c = qgen.generate(gold, seed=8)

    assert [q.id for q in a] == [q.id for q in b]
    assert [q.text for q in a] == [q.text for q in b]
    assert [q.id for q in a] != [q.id for q in c]


def test_no_unbound_slot_survives_generation(gold):
    for q in qgen.generate(gold, seed=1):
        assert "{" not in q.text, q.text
        assert "}" not in q.text, q.text


def test_ids_are_unique(gold):
    ids = [q.id for q in qgen.generate(gold, seed=1)]
    assert len(set(ids)) == len(ids)


def test_entities_are_recorded_so_epochs_can_be_kept_disjoint(gold):
    for q in qgen.generate(gold, seed=1):
        # Every question names at least one person or unit; the driver needs
        # these to keep entity pools disjoint across epochs, which is what stops
        # a memorised fact about one person from helping a later question.
        assert q.entities, q.id


def test_all_questions_in_a_basket_have_distinct_text(gold):
    # Checked across several seeds, not one: a single seed that happens not to
    # collide would pass for the wrong reason. `_pick()` draws independently
    # per index, which is sampling *with* replacement from a finite pool — two
    # different indices in the same class can land on the same unit (or the
    # same person, or the same unit+grade pair) and emit the exact same
    # Russian sentence twice. A duplicated question does not add information
    # to the basket; it double-weights one draw and shrinks the effective
    # sample the experiment is trying to resolve small differences with.
    for seed in (1, 2, 7, 8, 42):
        texts = [q.text for q in qgen.generate(gold, seed=seed)]
        duplicates = [t for t in set(texts) if texts.count(t) > 1]
        assert not duplicates, (seed, duplicates[:3])
