"""Тесты корпуса: инварианты, детерминизм и работоспособность API.

Корпус для тестов собирается один раз на сессию в каталоге ``.pytest-data``:
сборка на 800 человек занимает секунды, а зависимость от заранее собранных
данных сделала бы тесты нечестными.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heimdall.catalog import Catalog                       # noqa: E402
from heimdall.engine.execute import execute                # noqa: E402
from heimdall.engine.quirks import Quirks                  # noqa: E402

from b2e.build import build                                # noqa: E402
from b2e.store import ProceduralSnapshot                   # noqa: E402
from b2e.validate import run as validate_run               # noqa: E402

CATALOG = ROOT / "catalog" / "snapshot.json"
DATA = ROOT / ".pytest-data"
SEED = 424242


@pytest.fixture(scope="session")
def corpus():
    shutil.rmtree(DATA, ignore_errors=True)
    build(SEED, 800, DATA, CATALOG, progress=False)
    yield DATA
    shutil.rmtree(DATA, ignore_errors=True)


@pytest.fixture(scope="session")
def snap(corpus):
    return ProceduralSnapshot(corpus, Catalog.load(CATALOG))


@pytest.fixture(scope="session")
def catalog():
    return Catalog.load(CATALOG)


# ------------------------------------------------------------------ каталог

def test_catalog_matches_documented_inventory(catalog):
    counts = catalog.counts()
    assert counts == {"models": 37, "schemas": 7, "columns": 4599, "metrics": 148}
    assert len(catalog.history_models()) == 4


# ----------------------------------------------------------------- инварианты

def test_consistency_gate_passes(corpus):
    report = validate_run(corpus, CATALOG)
    failed = [row for row in report.rows if row[0] == "FAIL"]
    assert not failed, report.render()


def test_every_mart_has_rows_except_declared_empty(snap):
    from b2e.gen.marts import EMPTY_BY_DESIGN
    empty = {key for key in snap.keys() if snap.rows(key) == 0}
    assert empty == EMPTY_BY_DESIGN


def test_identity_is_shared_across_marts(snap):
    base = snap.table("dm_core.employee_actual")
    names = dict(zip(base.column("person_id"), base.column("employee_full_name")))
    other = snap.table("dm_special.employee_competence_actual")
    pairs = list(zip(other.column("person_id"), other.column("employee_full_name")))
    assert pairs and all(names[pid] == name for pid, name in pairs if pid in names)


def test_competencies_differ_within_person(snap):
    t = snap.table("dm_special.employee_competence_actual")
    cols = [t.column(c) for c in ("soft_skills_competency_score",
                                  "management_competency_score",
                                  "cognitive_features_group_score")]
    spread = [max(row) - min(row) for row in zip(*cols)]
    assert float(np.mean(spread)) > 0.3


def test_ability_reaches_observable_columns(snap, corpus):
    truth = json.loads((Path(corpus) / "truth" / "people.json").read_text("utf-8"))
    ability = dict(zip(truth["person_id"], truth["ability"]))
    t = snap.table("dm_core.employee_actual")
    pid = t.column("person_id")
    grade = t.column("grade_level")
    pairs = [(ability[p], g) for p, g in zip(pid, grade) if p in ability]
    x, y = zip(*pairs)
    assert 0.35 < float(np.corrcoef(x, y)[0, 1]) < 0.75


def test_latent_factors_are_not_served(snap):
    for key in snap.keys():
        available = snap.table(key).available()
        assert not [c for c in available if c.endswith("_hidden")]
        assert "ability" not in available


# --------------------------------------------------------------- детерминизм

def test_build_is_deterministic(tmp_path):
    first = build(SEED, 400, tmp_path / "a", CATALOG, progress=False)
    second = build(SEED, 400, tmp_path / "b", CATALOG, progress=False)
    assert first["snapshot_id"] == second["snapshot_id"]


def test_procedural_columns_are_stable(snap):
    t = snap.table("dm_core.employee_actual")
    tail = [c for c in t.available() if c not in t._index][:1]
    assert tail, "ожидается непустой процедурный хвост"
    first = t.column(tail[0])
    t._cache.clear()
    assert first == t.column(tail[0])


# ---------------------------------------------------------------------- API

def test_rows_query(snap, catalog):
    model = catalog.models["dm_core.employee_actual"]
    body = {"schema": "dm_core", "logic_model": "employee_actual",
            "columns": ["employee_full_name", "grade_level", "city_name"],
            "filters": {"type": "condition", "column": "employee_status",
                        "operator": "=", "value": "активный"},
            "limit": 5}
    out = execute(body, model, snap.table("dm_core.employee_actual"), Quirks())
    assert len(out["data"]) == 5
    assert all(row["employee_full_name"] for row in out["data"])


def test_response_is_json_serialisable(snap, catalog):
    model = catalog.models["dm_core.employee_actual"]
    body = {"schema": "dm_core", "logic_model": "employee_actual",
            "columns": ["employee_full_name", "educ.item_name", "goals.progress",
                        "educational_institution_name"],
            "limit": 3}
    out = execute(body, model, snap.table("dm_core.employee_actual"), Quirks())
    json.dumps(out, ensure_ascii=False)          # падение здесь = numpy в ответе


def test_unknown_column_is_rejected(snap, catalog):
    from heimdall.engine.errors import HeimdallError
    model = catalog.models["dm_core.employee_actual"]
    body = {"schema": "dm_core", "logic_model": "employee_actual",
            "columns": ["no_such_column_at_all"], "limit": 1}
    with pytest.raises(HeimdallError):
        execute(body, model, snap.table("dm_core.employee_actual"), Quirks())


def test_history_panel(snap, catalog):
    model = catalog.models["dm_core.employee_hist"]
    body = {"schema": "dm_core", "logic_model": "employee_hist",
            "columns": ["grade_level"],
            "time_dimensions": [{"name": "report_date", "granularity": "quarter_end",
                                 "range": {"type": "absolute", "from": "2025-01-01",
                                           "to": "2026-01-01"}}],
            "limit": 12}
    out = execute(body, model, snap.table("dm_core.employee_hist"), Quirks())
    assert out["data"], "панель истории пуста"
    assert {"report_date", "person_id"} <= set(out["data"][0])


# ------------------------------------------------------------------ словари

def test_names_have_gender_agreement(snap):
    t = snap.table("dm_core.employee_actual")
    full = t.column("employee_full_name")
    last = t.column("employee_last_name")
    assert all(f.split()[0].casefold() == l.casefold() for f, l in zip(full, last))


def test_universities_are_plausible(snap):
    t = snap.table("dm_core.employee_actual")
    flat = [v for row in t.column("educational_institution_name") for v in (row or [])]
    assert len(set(flat)) > 40
    assert all(len(name) > 2 and "institute_name" not in name for name in flat)
