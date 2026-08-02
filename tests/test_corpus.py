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


# ------------------------------------------------- вложенные массивы (группы)

SUCCESSOR_GROUP = ["successors.person_id", "successors.employee_id",
                   "successors.full_name", "successors.status",
                   "successors.appoint_date"]
PREDECESSOR_GROUP = ["predecessor.person_id", "predecessor.employee_id",
                     "predecessor.full_name", "predecessor.positions_id",
                     "predecessor.positions_name", "predecessor.status",
                     "predecessor.appoint_date"]


def test_successor_arrays_are_aligned_with_qty(snap):
    """У одного человека все successors.* одной длины, равной successors_qty."""
    t = snap.table("dm_core.employee_actual")
    arrays = {c: t.column(c) for c in SUCCESSOR_GROUP}
    qty = t.column("successors_qty")
    assert sum(qty) > 0, "в корпусе нет ни одного преемника"
    for i, q in enumerate(qty):
        lengths = {len(arrays[c][i] or []) for c in SUCCESSOR_GROUP}
        assert lengths == {int(q)}, (i, q, lengths)


def test_nobody_is_their_own_successor(snap):
    """«Кто может меня заменить» не должно отвечать «вы сами».

    Розыгрыш брал кандидата из всего блока, включая носителя позиции, поэтому
    один руководитель из 188 на выборке в 3000 человек оказывался в собственном
    резерве. С точки зрения агента это не странность данных, а неверный ответ на
    прямо заданный вопрос.
    """
    t = snap.table("dm_core.employee_actual")
    for own, successors in zip(t.column("employee_id"),
                               t.column("successors.employee_id")):
        assert str(own) not in {str(x) for x in (successors or [])}, own


def test_a_successor_is_named_once_per_person(snap):
    """Резерв из двух человек — это два человека, а не один, названный дважды.

    Бросок на каждый элемент списка был независимым, и с вероятностью 1/span
    два элемента совпадали. Теперь бросок один на человека, а элементы берут
    последовательные позиции пула, так что совпасть они не могут.
    """
    t = snap.table("dm_core.employee_actual")
    for own, successors in zip(t.column("employee_id"),
                               t.column("successors.employee_id")):
        ids = [str(x) for x in (successors or [])]
        assert len(ids) == len(set(ids)), (own, ids)


def test_successors_qty_never_exceeds_the_available_pool(snap):
    """В блоке из одного человека преемника нет и быть не может.

    Счётчик обязан это признать, а не пообещать резерв, под который в блоке нет
    кандидатов: пустая строка при qty=1 читается как потеря данных.
    """
    t = snap.table("dm_core.employee_actual")
    for qty, ids in zip(t.column("successors_qty"),
                        t.column("successors.employee_id")):
        assert int(qty or 0) == len(ids or []), (qty, ids)


def test_successor_identity_is_real(snap):
    """person_id / employee_id / full_name преемника ищутся в других витринах."""
    from b2e.gen.dicts import GENERIC_WORDS
    t = snap.table("dm_core.employee_actual")
    by_person = dict(zip(t.column("person_id"), t.column("employee_full_name")))
    by_employee = dict(zip((str(v) for v in t.column("employee_id")),
                           t.column("employee_full_name")))
    seen = 0
    for pids, eids, fios in zip(t.column("successors.person_id"),
                                t.column("successors.employee_id"),
                                t.column("successors.full_name")):
        for pid, eid, fio in zip(pids or [], eids or [], fios or []):
            seen += 1
            # Ровно тот дефект, ради которого всё это: раньше здесь лежало
            # «итоговый» из словаря заполнителя, а employee_id ни с чем не
            # соединялся.
            assert fio not in GENERIC_WORDS
            assert by_person[pid] == fio
            assert by_employee[str(eid)] == fio
    assert seen > 0


def test_predecessor_is_the_inverse_of_successors(snap):
    """predecessor.* — та же связь с другой стороны, а не копия successors.*."""
    from collections import Counter
    t = snap.table("dm_core.employee_actual")
    subject = [str(v) for v in t.column("employee_id")]
    forward = Counter(
        (who, str(e), s)
        for who, eids, sts in zip(subject, t.column("successors.employee_id"),
                                  t.column("successors.status"))
        for e, s in zip(eids or [], sts or []))
    backward = Counter(
        (str(e), who, s)
        for who, eids, sts in zip(subject, t.column("predecessor.employee_id"),
                                  t.column("predecessor.status"))
        for e, s in zip(eids or [], sts or []))
    assert forward and forward == backward
    # positions_* есть только у predecessor: это позиция замещаемого.
    position = dict(zip(subject, t.column("position_id")))
    pairs = [(str(e), p) for eids, pos in zip(t.column("predecessor.employee_id"),
                                              t.column("predecessor.positions_id"))
             for e, p in zip(eids or [], pos or [])]
    assert pairs and all(position[e] == p for e, p in pairs)


def test_predecessor_arrays_are_aligned(snap):
    t = snap.table("dm_core.employee_actual")
    arrays = {c: t.column(c) for c in PREDECESSOR_GROUP}
    assert any(arrays[PREDECESSOR_GROUP[0]])
    for i in range(t.rows):
        assert len({len(arrays[c][i] or []) for c in PREDECESSOR_GROUP}) == 1


def test_predecessor_answers_on_orion_where_successors_absent(snap, catalog):
    """Витрина orion объявляет только predecessor.* — и он там содержателен.

    Это и есть довод в пользу обратного прочтения: при зеркальном значении
    технологический блок (почти сплошь не руководители) дал бы пустую витрину.
    """
    key = "dm_core.employee_actual_orion"
    assert not [c for c in catalog.models[key].columns if c.startswith("successors.")]
    t = snap.table(key)
    filled = sum(1 for v in t.column("predecessor.employee_id") if v)
    assert filled > 0


def test_nested_groups_have_one_length_per_row(snap):
    """Классовый инвариант: члены любой смысловой группы массивов равны по длине."""
    from b2e.validate import NESTED_MARTS, _nested_groups
    checked = 0
    for key in NESTED_MARTS:
        if not snap.has(key):
            continue
        table = snap.table(key)
        model = snap.catalog.models[key]
        for group, cols in sorted(_nested_groups(model, table).items()):
            values = {c: table.column(c) for c in cols}
            for i in range(table.rows):
                assert len({len(values[c][i] or []) for c in cols}) == 1, \
                    f"{key}.{group} строка {i}"
            table._cache.clear()
            checked += 1
    assert checked >= 8


def test_successor_arrays_survive_the_api(snap, catalog):
    """Форма, которую видит агент: два члена группы в одном ответе."""
    model = catalog.models["dm_core.employee_actual"]
    body = {"schema": "dm_core", "logic_model": "employee_actual",
            "columns": ["employee_id", "successors_qty", "successors.full_name",
                        "successors.employee_id", "successors.status"],
            "limit": 300}
    out = execute(body, model, snap.table("dm_core.employee_actual"), Quirks())
    json.dumps(out, ensure_ascii=False)
    rows = [r for r in out["data"] if r["successors.employee_id"]]
    assert rows, "в выборке нет ни одного человека с преемниками"
    for row in rows:
        assert (len(row["successors.full_name"])
                == len(row["successors.employee_id"])
                == len(row["successors.status"])
                == int(row["successors_qty"]))


def test_gate_fails_when_successor_identity_falls_back_to_filler(corpus, tmp_path):
    """Гейт обязан падать ровно на том дефекте, ради которого заведён.

    Дефект был не в раскладке, а в том, что колонки личности вообще не писались
    и на чтении доставались ``fillers.column``. Пока условие проверки читалось с
    ``table._index``, возврат этого дефекта не ронял гейт, а бесшумно уносил с
    собой сами проверки: 42 → 38, «отказов: 0», при том что
    ``successors.full_name`` снова содержал «итоговый». Здесь эти колонки
    убираются из индекса копии корпуса — так же, как их не было до починки.
    """
    strip = ["successors.person_id", "successors.employee_id",
             "successors.full_name", "predecessor.person_id",
             "predecessor.employee_id", "predecessor.full_name",
             "predecessor.positions_id", "predecessor.positions_name"]
    broken = tmp_path / "broken"
    shutil.copytree(corpus, broken)
    for mart in broken.iterdir():
        index = mart / "_index.json"
        if not index.is_file():
            continue
        payload = json.loads(index.read_text("utf-8"))
        dropped = [c for c in strip if payload["columns"].pop(c, None) is not None]
        if dropped:
            index.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_run(broken, CATALOG)
    names = [row[1] for row in report.rows if row[0] == "FAIL"]
    assert "W7 личность преемника настоящая" in names, report.render()
    assert "W7 predecessor обратен successors" in names, report.render()


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
