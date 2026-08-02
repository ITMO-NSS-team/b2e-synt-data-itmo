"""Сборка корпуса: популяция → блоки → проекции витрин → снимок на диске.

Что здесь важно понимать
------------------------
Порядок сборки — это и есть модель предметной области: сначала организация,
потом люди в ней, потом события с людьми, и только потом витрины как проекции.
Обратный порядок (витрина сама придумывает себе людей) — ровно та ошибка, из-за
которой в прежнем корпусе один ``person_id`` носил разные имена на разных
витринах.

Истина оценки (латентные факторы, gold-метки) пишется в отдельный каталог
``truth/`` и API не отдаётся никогда: иначе агент прочитает ответ вместо того,
чтобы его вывести.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from heimdall.catalog import Catalog

from b2e import store as store_mod
from b2e import traps
from b2e.gen import blocks as blocks_mod
from b2e.gen import dicts, marts, population
from b2e.gen.resolve import Resolver, person_uuid

#: Служебные колонки истории: список изменений ``(дата, колонка, значение)``.
CHANGES_COLUMN = "_changes"

#: Колонки, к которым применяется регистровая каверза.
_NAME_COLUMNS = {"employee_full_name", "full_name"}

#: Вложенные массивы ФИО: колонка → блок, чьё поле ``row`` даёт человека для
#: каждого элемента. Каверза привязана к человеку, поэтому ей нужен именно он.
_NESTED_NAME_COLUMNS = {"successors.full_name": "successors",
                        "predecessor.full_name": "predecessors"}


def build(seed: int, n_people: int, out: str | Path,
          catalog_path: str | Path = "catalog/snapshot.json",
          progress: bool = True, enabled_traps: set[str] | None = None) -> dict:
    """Собрать корпус целиком и вернуть сводку сборки."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = Catalog.load(catalog_path)
    enabled_traps = (traps.DEFAULT_ENABLED if enabled_traps is None else enabled_traps)

    t0 = time.time()
    pop = population.build(seed, n_people)
    _log(progress, f"популяция: {pop.n} человек, {len(pop.tree)} подразделений "
                   f"({time.time() - t0:.1f}s)")

    t1 = time.time()
    blocks = blocks_mod.build_all(pop)
    _log(progress, f"блоки: " + ", ".join(
        f"{k}={b.total}" for k, b in blocks.items()) + f" ({time.time() - t1:.1f}s)")

    resolver = Resolver(pop, blocks, seed)
    tables: dict[str, str] = {}
    stats: dict[str, dict] = {}

    for key in sorted(catalog.models):
        model = catalog.models[key]
        t2 = time.time()
        frame = marts.frame_for(key, pop, resolver)
        writer = store_mod.ChunkedWriter(out, key)
        written = 0
        if frame is not None and frame.n > 0:
            for name, member in model.columns.items():
                if member.kind == "param_column":
                    continue                      # считается на запросе из аргументов
                values = resolver.column(key, member, frame, strict=True)
                if values is None:
                    continue                      # хвост — процедурно на чтении
                if name in _NAME_COLUMNS:
                    values = traps.apply_upper(values, seed, frame.person,
                                               "upper_cyrillic" in enabled_traps)
                elif name in _NESTED_NAME_COLUMNS:
                    refs = blocks[_NESTED_NAME_COLUMNS[name]].lists(
                        "row", np.maximum(frame.person, 0))
                    values = traps.apply_upper_nested(
                        values, seed, refs, int(pop.n),
                        "upper_cyrillic" in enabled_traps)
                writer.write(name, values, member.type)
                written += 1
            if model.is_history:
                writer.write(CHANGES_COLUMN, _history(pop, frame), "internal")
        tables[key] = writer.close()
        stats[key] = {"rows": frame.n if frame else 0, "materialised": written,
                      "declared": len(model.columns),
                      "seconds": round(time.time() - t2, 2)}
        _log(progress, f"  {key:<52} строк={stats[key]['rows']:>7} "
                       f"колонок={written:>4}/{len(model.columns)}")

    snapshot_id = store_mod.manifest(
        out, seed=seed, catalog=catalog, tables=tables,
        extra={"people": int(pop.n), "units": int(len(pop.tree)),
               "catalog_path": str(catalog_path), "as_of": str(population.AS_OF),
               "traps": sorted(enabled_traps)})
    _write_truth(pop, blocks, resolver, out)
    summary = {"snapshot_id": snapshot_id, "people": int(pop.n),
               "units": int(len(pop.tree)), "seconds": round(time.time() - t0, 1),
               "tables": stats}
    (out / "_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(progress, f"снимок {snapshot_id} за {summary['seconds']}s")
    return summary


def _history(pop, frame) -> list:
    """Событийная история: изменения, которые сворачиваются в текущее состояние.

    Инвариант: применение всех событий к базовой строке даёт ровно то, что лежит
    в ``employee_actual``. В прежнем корпусе базовая строка уже была текущим
    состоянием, а события лежали рядом и ни во что не сворачивались.
    """
    from b2e.gen.population import AS_OF_DAYS, _days_to_iso
    rows = frame.person
    out: list = []
    grade = pop["grade_level"]
    start = pop["position_start_days"]
    hire = pop["hire_days"]
    fired = pop["fired_days"]
    for r in rows:
        if r < 0:
            out.append([])
            continue
        events = []
        if start[r] > hire[r] + 200:              # был перевод/повышение
            events.append([_days_to_iso(np.array([start[r]]))[0],
                           "grade_level", int(grade[r])])
        if fired[r] > 0:
            day = _days_to_iso(np.array([fired[r]]))[0]
            events.append([day, "employee_status", "уволен"])
            events.append([day, "fact_flag", 0])
        out.append(events)
    return out


def _write_truth(pop, blocks, resolver, out: Path) -> None:
    """Истина для оценки агента. Не витрина и через API недоступна."""
    truth = out / "truth"
    truth.mkdir(parents=True, exist_ok=True)
    rows = np.arange(pop.n)
    payload = {
        "person_id": person_uuid(pop, rows).tolist(),
        "employee_id": [int(v) for v in pop["employee_id"]],
        "ability": np.round(pop.truth["ability"], 4).tolist(),
        "potential": np.round(pop.truth["potential"], 4).tolist(),
        "perf_latent": np.round(pop.truth["perf_latent"], 4).tolist(),
        "competency_avg": np.round(pop.truth["competency_avg"], 4).tolist(),
        "attrition_risk": [int(v) for v in pop["attrition_risk_idx"]],
        "is_head": [int(v) for v in pop["is_head"]],
        "unit_id": [int(v) for v in pop["unit_id"]],
        "grade_level": [int(v) for v in pop["grade_level"]],
    }
    (truth / "people.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (truth / "README.md").write_text(
        "# Истина оценки\n\n"
        "Скрытые факторы и gold-метки. Через API Heimdall недоступны и в витрины\n"
        "не попадают: агент должен выводить ответ из данных, а не читать его.\n"
        "Используется только скорером симуляции.\n", encoding="utf-8")


def _log(on: bool, message: str) -> None:
    if on:
        print(message, flush=True)
