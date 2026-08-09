"""Reference answers are computed from the population, not from the marts.

Every test here builds a small corpus and asserts against numpy arrays taken
straight from ``truth/people.json``. That duplication is deliberate: if the
reference evaluator and the test both went through the same helper, a bug in
the helper would agree with itself and the test would pass.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b2e.build import build                                    # noqa: E402
from sim.oracle.labels import GoldLabels, ROUND_DP             # noqa: E402
from sim.oracle import reference as ref                        # noqa: E402

CATALOG = ROOT / "catalog" / "snapshot.json"
DATA = ROOT / ".pytest-reference"
SEED = 424242


@pytest.fixture(scope="module")
def gold():
    shutil.rmtree(DATA, ignore_errors=True)
    build(SEED, 800, DATA, CATALOG, progress=False)
    yield GoldLabels(DATA)
    shutil.rmtree(DATA, ignore_errors=True)


def _some_unit(gold) -> int:
    """A unit with at least 20 people, so counts and medians are not degenerate.

    Scans every unit in the tree, not just the leaves that hold people
    directly (``set(gold.unit_of)``): on the 800-person test corpus the org
    tree is deep enough (163 units, 5 levels) that no leaf alone reaches 20
    members, but an internal unit's *recursive* membership does. Restricting
    the scan to directly-populated units would raise on this corpus even
    though plenty of qualifying scopes exist.
    """
    for unit_id in range(len(gold.tree)):
        if len(gold.members(unit_id)) >= 20:
            return unit_id
    raise AssertionError("no unit with >=20 members in the test corpus")


def test_count_with_predicate(gold):
    unit = _some_unit(gold)
    spec = ref.ReferenceSpec(
        op="count", field="grade_level",
        scope={"unit_id": unit, "recursive": True},
        predicate={"op": ">=", "value": 12}, n=None, refs=(),
    )
    rows = gold.members(unit)
    expected = int((gold.grade_level[rows] >= 12).sum())

    result = ref.evaluate(spec, gold)

    assert result.kind == "number"
    assert result.value == expected


def test_median_rounds_to_round_dp(gold):
    unit = _some_unit(gold)
    spec = ref.ReferenceSpec(
        op="median", field="performance_pct",
        scope={"unit_id": unit, "recursive": True},
        predicate=None, n=None, refs=(),
    )
    rows = gold.members(unit)
    expected = round(float(np.median(gold.performance_pct[rows])), ROUND_DP)

    result = ref.evaluate(spec, gold)

    assert result.kind == "number"
    assert result.value == expected


def test_top_n_returns_ids_in_rank_order(gold):
    unit = _some_unit(gold)
    spec = ref.ReferenceSpec(
        op="top_n", field="potential_pct",
        scope={"unit_id": unit, "recursive": True},
        predicate=None, n=3, refs=(),
    )
    rows = gold.members(unit)
    order = rows[np.argsort(-gold.potential_pct[rows], kind="stable")][:3]
    expected = tuple(gold.person_id[int(r)] for r in order)

    result = ref.evaluate(spec, gold)

    assert result.kind == "ids"
    assert result.ids == expected


def test_matches_is_order_sensitive_for_rankings(gold):
    r = ref.Reference(kind="ids", value=None, ids=("a", "b", "c"), verdict=None)

    assert ref.matches(r, value=None, ids=["a", "b", "c"], verdict=None)
    assert not ref.matches(r, value=None, ids=["b", "a", "c"], verdict=None)
    assert not ref.matches(r, value=None, ids=["a", "b"], verdict=None)


def test_matches_tolerates_float_noise_within_round_dp(gold):
    r = ref.Reference(kind="number", value=41.2345, ids=(), verdict=None)

    assert ref.matches(r, value=41.23450001, ids=None, verdict=None)
    assert not ref.matches(r, value=41.2346, ids=None, verdict=None)


def test_unknown_op_raises_rather_than_returning_none(gold):
    spec = ref.ReferenceSpec(op="nonsense", field="grade_level", scope={},
                             predicate=None, n=None, refs=())

    with pytest.raises(ValueError, match="nonsense"):
        ref.evaluate(spec, gold)


def test_reference_module_does_not_import_the_query_engine():
    """The reference must not go through the projection the agent queries.

    Enforced by parsing rather than by convention: a future edit that reaches
    for ``heimdall.engine.execute`` to save effort would make every reference
    agree with the mart, including where the mart is wrong.
    """
    import ast

    src = (ROOT / "sim" / "oracle" / "reference.py").read_text("utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name.startswith("heimdall") for name in imported), imported
