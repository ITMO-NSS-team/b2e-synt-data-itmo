"""Гейт стоимости запроса: отказ до аллокации вместо OOM после неё.

Проверяется не только сам отказ, но и два расхождения, которые сделали бы гейт
бесполезным незаметно:

* множество колонок, по которому считается бюджет, обязано совпадать с тем,
  которое потом действительно читается;
* опознание эмбеддингов здесь — дубликат правила из генератора, и разъехаться
  они не должны.
"""
from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from heimdall.catalog import Catalog
from heimdall.engine import budget as budget_mod
from heimdall.engine.budget import (
    ARRAY_BYTES_PER_ROW, EMBEDDING_BYTES_PER_ROW, SCALAR_BYTES_PER_ROW,
    Budget, estimate_bytes,
)
from heimdall.engine.errors import ERRORS, HeimdallError

CATALOG = pathlib.Path("catalog/snapshot.json")
HAS_CORPUS = pathlib.Path("data-small/manifest.json").exists()
BEARER = {"Authorization": "Bearer test-token"}

WIDE = "dm_core.employee_actual"


@pytest.fixture(scope="module")
def catalog():
    return Catalog.load(CATALOG)


@pytest.fixture(scope="module")
def model(catalog):
    return catalog.models[WIDE]


# ------------------------------------------------------------------- оценка

def test_scalar_column_priced_per_row(model):
    name = next(n for n, m in model.columns.items()
                if not m.ch.is_array and not budget_mod.is_embedding(n))
    assert estimate_bytes(model, [name], 1000) == 1000 * SCALAR_BYTES_PER_ROW


def test_array_column_costs_more_than_scalar(model):
    array = next(n for n, m in model.columns.items()
                 if m.ch.is_array and not budget_mod.is_embedding(n))
    scalar = next(n for n, m in model.columns.items()
                  if not m.ch.is_array and not budget_mod.is_embedding(n))
    assert (estimate_bytes(model, [array], 1000)
            == 1000 * ARRAY_BYTES_PER_ROW > estimate_bytes(model, [scalar], 1000))


def test_embedding_dominates_every_other_kind(catalog):
    """Вектор на 384 измерения дороже скаляра на два порядка. Бюджет в ячейках
    этого не увидел бы — потому бюджет и в байтах."""
    model = catalog.models["anagent.oss_ebase_embs"]
    name = next(n for n in model.columns if budget_mod.is_embedding(n))
    assert estimate_bytes(model, [name], 100) == 100 * EMBEDDING_BYTES_PER_ROW
    assert EMBEDDING_BYTES_PER_ROW > 100 * SCALAR_BYTES_PER_ROW


def test_empty_table_costs_nothing(model):
    assert estimate_bytes(model, list(model.columns)[:50], 0) == 0


def test_unknown_column_is_not_priced(model):
    """Чужую ошибку гейт не подменяет: неизвестное имя — дело unknown-column."""
    assert estimate_bytes(model, ["нет-такой-колонки"], 10_000) == 0


def test_cost_grows_with_column_count(model):
    names = [n for n in list(model.columns)[:40]]
    assert (estimate_bytes(model, names[:20], 5000)
            < estimate_bytes(model, names, 5000))


# -------------------------------------------------------------------- отказ

def test_enforce_passes_within_budget(model):
    names = list(model.columns)[:5]
    budget_mod.enforce(model, names, 3000, Budget(1024 ** 3))


def test_enforce_refuses_over_budget(model):
    names = list(model.columns)[:200]
    with pytest.raises(HeimdallError) as exc:
        budget_mod.enforce(model, names, 294_000, Budget(64 * 1024 ** 2))
    assert exc.value.code == "query-too-expensive"
    assert exc.value.status == 422


def test_zero_budget_disables_the_gate(model):
    budget_mod.enforce(model, list(model.columns), 10 ** 9, Budget.off())


def test_verdict_is_deterministic(model):
    """Один и тот же запрос — один и тот же вердикт: на этом держатся
    replay-фикстуры и кэш прогонов."""
    names = list(model.columns)[:200]
    first = estimate_bytes(model, names, 294_000)
    for _ in range(5):
        assert estimate_bytes(model, names, 294_000) == first


def test_error_is_registered_with_a_hint():
    spec = ERRORS["query-too-expensive"]
    assert spec.status == 422
    # Подсказка обязана предупредить, что limit не поможет: он и правда не
    # уменьшает стоимость, и совет «возьми limit поменьше» увёл бы агента в
    # тупик, из которого он бы не вышел.
    assert "limit" in spec.hint


def test_detail_names_the_numbers_the_agent_needs(model):
    with pytest.raises(HeimdallError) as exc:
        budget_mod.enforce(model, list(model.columns)[:300], 294_000,
                           Budget(32 * 1024 ** 2))
    detail = exc.value.detail
    assert "294000" in detail and WIDE in detail and "МБ" in detail


# ---------------------------------------------------------- против расхождения

def test_embedding_rule_matches_the_generator(catalog):
    """Правило продублировано из b2e.gen.fillers — разъехаться им нельзя."""
    fillers = pytest.importorskip("b2e.gen.fillers")
    names = {n for m in catalog.models.values() for n in m.columns}
    assert names
    mismatched = [n for n in names
                  if budget_mod.is_embedding(n) != fillers.is_embedding(n)]
    assert not mismatched


def test_budget_counts_exactly_what_prefetch_reads(model):
    """Гейт, который меряет не то множество, которое читается, хуже отсутствия
    гейта. Здесь оба берутся из одной функции — проверяется, что это так и
    осталось."""
    from heimdall.engine.compile import compile_query
    from heimdall.engine.execute import _metric_specs, referenced_column_names
    from heimdall.engine.metrics import DEFAULT_REGISTRY

    names = [n for n in sorted(model.selectable_names())][:4]
    order = names[0]
    body = {"schema": model.schema, "logic_model": model.logic_model,
            "columns": names,
            "order_by": [{"field": order, "kind": "column", "direction": "asc"}]}
    plan = compile_query(body, model, None)
    specs = _metric_specs(plan, DEFAULT_REGISTRY)
    wanted = referenced_column_names(plan, specs)

    touched: list[str] = []

    class Recorder:
        rows = 10

        def column(self, name):
            touched.append(name)
            return [None] * self.rows

    from heimdall.engine.execute import _prefetch
    _prefetch(plan, lambda n: Recorder().column(n), specs)
    assert set(touched) == wanted


# ------------------------------------------------------------- сквозь HTTP

def _client(budget_bytes: int | None = None):
    from sim.emulator.app import create_app
    from sim.emulator.config import EmulatorConfig

    kw = {} if budget_bytes is None else {"query_budget_bytes": budget_bytes}
    return TestClient(create_app(EmulatorConfig(
        snapshot_traps_on="data-small", latency_profile="instant", **kw)))


def _headers(client) -> dict:
    """Запрос всегда идёт от лица конкретного сотрудника; анонимного доступа нет."""
    actor = client.get("/control/identities?n=1").json()[0]
    return {**BEARER, "x-employee-id": str(actor["employee_id"])}


@pytest.mark.skipif(not HAS_CORPUS, reason="needs data-small; run `make seed`")
def test_wide_query_is_refused_over_http():
    client = _client(budget_bytes=4096)
    model = Catalog.load(CATALOG).models[WIDE]
    body = {"schema": model.schema, "logic_model": model.logic_model,
            "columns": list(sorted(model.selectable_names()))[:100], "limit": 1}
    r = client.post("/api/v1/mcp/query/", json=body, headers=_headers(client))
    assert r.status_code == 422, r.text
    payload = r.json()
    assert payload["code"] == "query-too-expensive"
    assert payload["hint"] and payload["detail"]


@pytest.mark.skipif(not HAS_CORPUS, reason="needs data-small; run `make seed`")
def test_narrow_query_still_works():
    client = _client()
    model = Catalog.load(CATALOG).models[WIDE]
    body = {"schema": model.schema, "logic_model": model.logic_model,
            "columns": list(sorted(model.selectable_names()))[:3], "limit": 5}
    r = client.post("/api/v1/mcp/query/", json=body, headers=_headers(client))
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) <= 5
