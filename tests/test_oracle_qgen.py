"""The generated half of the basket.

Two properties matter more than the text: every question's reference must
evaluate without raising, and generation must be a pure function of the seed.
A generator that occasionally emits an unanswerable question would put a
permanent floor under every arm's score, and one that is not reproducible
would make a run impossible to re-score later.
"""
from __future__ import annotations

import json
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


def test_every_unit_scoped_question_states_that_the_subtree_counts(gold):
    """The reference always evaluates recursively; the text never said so.

    On the 800-person corpus 95 of the 120 named units have zero direct
    members — they are internal org nodes — and 92 of 120 questions get a
    different answer under a direct-membership reading. That is not a hard
    question. It is one sentence of learnable convention, and a single memory
    item that discovers it flips ~92 questions at once, against an expected
    between-arm effect of a few percentage points. The spec's anti-leakage
    machinery protects against memorising *facts*; nothing protects against
    memorising the scorer's *conventions* except stating them in the question.
    """
    for seed in (1, 2, 7):
        for q in qgen.generate(gold, seed=seed):
            if q.spec.scope.get("unit_id") is None:
                continue
            assert q.spec.scope.get("recursive") is True, q.id
            assert "включая подчинённые подразделения" in q.text, (q.id, q.text)


def test_every_reference_quantity_is_reachable_from_the_catalogue(gold):
    """A question whose answer no served column determines measures nothing.

    Two earlier classes ranked latents (`potential_pct`, `impact_pct`) that no
    mart exposes, and one ranked `performance_pct`, whose only served proxy is
    an A-E mark the generator deliberately makes noisy. Those are not hard
    questions either — they are a floor under every arm equally, and they would
    have spent 75 of the 180 deterministic slots on it. This test is what stops
    the basket from acquiring another one.
    """
    catalogue = json.loads((ROOT / "catalog" / "snapshot.json").read_text("utf-8"))

    def exists(path: str) -> bool:
        model, _, column = path.rpartition(".")
        entry = catalogue["models"].get(model)
        if entry is None:
            return False
        names = {c["name"] for c in entry.get("columns", ())}
        names |= {c["name"] for c in entry.get("metrics", ())}
        return column in names

    for path in qgen.UNIT_SCOPE_PATHS:
        assert exists(path), path
    for field, paths in qgen.MART_PATHS.items():
        assert paths, field
        for path in paths:
            assert exists(path), (field, path)

    for q in qgen.generate(gold, seed=1):
        assert q.spec.field in qgen.MART_PATHS, (q.id, q.spec.field)


def test_unit_scope_paths_reproduce_a_unit_scoped_answer_through_heimdall(gold):
    """`UNIT_SCOPE_PATHS` names the columns an agent would actually query.

    Being *in the catalogue* (the test above) is necessary but not sufficient
    — `oshs_level_N_unit_id_main` used to pass that test while being useless:
    it holds a mart-internal id space unrelated to `spec.scope["unit_id"]`,
    which is an index into the gold tree. An agent filtering on it would match
    nothing, on every one of the 105 unit-scoped questions. This test replays
    one such question through the real query engine, using only the columns
    the map names, and checks the result against the reference — the same
    check the re-reviewer ran by hand to find the bug.
    """
    from b2e.store import ProceduralSnapshot
    from heimdall.catalog import Catalog
    from heimdall.engine.execute import execute

    catalog = Catalog.load(CATALOG)
    model = catalog.models["dm_core.employee_actual"]
    reader = ProceduralSnapshot(DATA, catalog).table("dm_core.employee_actual")

    q = next(qq for qq in qgen.generate(gold, seed=1)
             if qq.question_class == "count_by_grade")
    unit_name = str(gold.tree.name[q.spec.scope["unit_id"]])
    grade = q.spec.predicate["value"]
    columns = [path.rsplit(".", 1)[1] for path in qgen.UNIT_SCOPE_PATHS]

    body = {
        "schema": model.schema, "logic_model": model.logic_model,
        "columns": ["person_id"],
        "filters": {"type": "and", "conditions": [
            {"type": "or", "conditions": [
                {"type": "condition", "column": col, "operator": "=",
                 "value": unit_name} for col in columns
            ]},
            {"type": "condition", "column": "grade_level", "operator": ">=",
             "value": grade},
        ]},
        "limit": 1000,
    }
    result = execute(body, model, reader)
    expected = ref.evaluate(q.spec, gold)

    assert expected.kind == "number"
    assert len(result["data"]) == expected.value, (q.text, unit_name)


def test_no_question_names_a_unit_whose_name_is_shared(gold):
    # A question names a unit in prose, not by id — "«Управление операционных
    # рисков»" is what reaches the agent. If a second unit anywhere in the org
    # tree renders the same name, the question has more than one correct
    # answer and `reference.evaluate` computes exactly one of them: a forced
    # wrong answer no matter how well the agent queried. The name-collision
    # count below is taken over the whole tree deliberately, not over the
    # generator's own eligible/unambiguous pools, so this test does not just
    # re-assert the generator's internal bookkeeping.
    name_counts: dict[str, int] = {}
    for u in range(len(gold.tree)):
        name = str(gold.tree.name[u])
        name_counts[name] = name_counts.get(name, 0) + 1

    for seed in (1, 2, 7):
        for q in qgen.generate(gold, seed=seed):
            for entity in q.entities:
                if entity.startswith("unit:"):
                    unit_id = int(entity.split(":", 1)[1])
                    name = str(gold.tree.name[unit_id])
                    assert name_counts[name] == 1, (seed, q.id, name)
