"""Режим истории: окно с шагом периода, carry-forward, панель и агрегат.

Почему история хранится событиями, а не снимками
------------------------------------------------
Снимок на каждую дату для 3 000 сотрудников × 443 колонки × 36 месяцев — это
48 млн ячеек, которые почти целиком повторяют друг друга. Событийная модель
хранит базовую строку и список изменений ``(дата, колонка, значение)``, а панель
на окно материализуется на лету. Это дешевле и точнее повторяет SCD2: значение
переносится вперёд ровно до следующего изменения.

Границы канала, которые обязан держать этот модуль:
* окно только абсолютное, ``[from, to)`` — правая граница исключается;
* ровно одна ось времени, ровно один элемент ``time_dimensions``;
* ширина ≤ ``default_lookback`` модели;
* в истории нет ``limit_by`` и нет параметрических членов;
* ``report_date`` не дублируется в ``columns`` — он и так в выдаче.

Первые четыре проверяются на этапе разбора тела (``compile.py``), здесь —
материализация и расчёт.
"""
from __future__ import annotations

import calendar
import datetime as dt
from typing import Any, Protocol

from .compile import Plan, TimeWindow
from .filters import evaluate
from .metrics import MetricSpec, compute
from .quirks import Quirks

#: Ключ сущности панели. У всех четырёх исторических моделей он один и тот же.
ENTITY_KEY = "person_id"

#: Колонка снимка со списком изменений: [(YYYY-MM-DD, колонка, значение), …].
CHANGES_COLUMN = "_changes"


class Reader(Protocol):
    rows: int

    def column(self, name: str) -> list[Any]: ...


# -------------------------------------------------------------------- периоды

def periods(window: TimeWindow) -> list[str]:
    """Правые границы периодов, попадающих в окно ``[from, to)``.

    Период включается, если его правая граница строго меньше ``to``. Поэтому
    окно ``[2026-01-01, 2026-01-31)`` при шаге ``month_end`` не даёт ни одного
    периода: конец января в него не попал.
    """
    start = dt.date.fromisoformat(window.date_from)
    end = dt.date.fromisoformat(window.date_to)
    if start >= end:
        return []

    out: list[str] = []
    cursor = start
    guard = 0
    while cursor < end:
        boundary = _period_end(cursor, window.granularity)
        if boundary < end:
            out.append(boundary.isoformat())
        cursor = boundary + dt.timedelta(days=1)
        guard += 1
        if guard > 20_000:  # pragma: no cover — защита от бесконечного цикла
            raise RuntimeError("периодов больше 20000: проверь гранулярность и окно")
    return out


def _period_end(day: dt.date, granularity: str) -> dt.date:
    if granularity == "day":
        return day
    if granularity == "week_end":
        # Неделя заканчивается воскресеньем: weekday() 0=пн … 6=вс.
        return day + dt.timedelta(days=6 - day.weekday())
    if granularity == "month_end":
        return dt.date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])
    if granularity == "quarter_end":
        last_month = ((day.month - 1) // 3 + 1) * 3
        return dt.date(day.year, last_month, calendar.monthrange(day.year, last_month)[1])
    if granularity == "year_end":
        return dt.date(day.year, 12, 31)
    raise ValueError(f"неизвестная гранулярность {granularity!r}")


# --------------------------------------------------------------- материализация

def materialize(reader: Reader, window: TimeWindow, columns: list[str],
                fill: str = "previous",
                keep: list[int] | None = None) -> list[dict]:
    """Развернуть событийную историю в панель «сущность × период».

    ``fill='previous'`` переносит последнее известное значение вперёд;
    ``fill='none'`` оставляет пропуск в периодах без изменений.
    """
    bounds = periods(window)
    if not bounds:
        return []

    rows = keep if keep is not None else list(range(reader.rows))
    entity = reader.column(ENTITY_KEY)
    base = {name: reader.column(name) for name in columns}
    changes = reader.column(CHANGES_COLUMN)

    out: list[dict] = []
    for i in rows:
        timeline = _timeline(changes[i], columns)
        current = {name: base[name][i] for name in columns}
        touched: set[str] = set()
        for boundary in bounds:
            fresh = set()
            for when, column, value in timeline:
                if when <= boundary and (column, when) not in touched:
                    current[column] = value
                    touched.add((column, when))
                    fresh.add(column)
            row = {"report_date": boundary, ENTITY_KEY: entity[i]}
            for name in columns:
                if fill == "none" and name not in fresh and boundary != bounds[0]:
                    row[name] = None
                else:
                    row[name] = current[name]
            out.append(row)
    return out


def _timeline(raw: Any, columns: list[str]) -> list[tuple[str, str, Any]]:
    """Отсортированный по дате список изменений интересующих нас колонок."""
    if not raw:
        return []
    wanted = set(columns)
    events = [(str(w), str(c), v) for w, c, v in raw if str(c) in wanted]
    return sorted(events, key=lambda e: e[0])


# ------------------------------------------------------------------ исполнение

def execute_history(plan: Plan, reader: Reader, quirks: Quirks, *,
                    query_type: str | None = None,
                    metrics_registry: dict | str) -> dict:
    """Посчитать историю: панель без метрик, динамику по периодам — с метриками."""
    window = plan.window
    assert window is not None  # гарантировано compile_query

    # Фильтр применяется к БАЗОВОЙ строке сущности: отбор идёт по тому, кто
    # попадает в панель, а не по значению внутри отдельного периода.
    mask = evaluate(plan.filters, plan.model, reader.rows,
                    lambda name: reader.column(name), quirks)
    kept = [i for i, ok in enumerate(mask) if ok]

    specs = _specs(plan, metrics_registry)
    needed = list(dict.fromkeys(plan.columns + sorted(
        {c for spec in specs.values() for c in _spec_columns(spec)})))
    panel = materialize(reader, window, needed, fill=window.fill, keep=kept)

    if not specs:
        rows = [{k: v for k, v in row.items()
                 if k in ("report_date", ENTITY_KEY) or k in plan.columns}
                for row in panel]
    else:
        rows = _aggregate_periods(panel, plan, specs)

    rows = _order(rows, plan)
    page = rows[plan.offset: plan.offset + plan.limit + 1]
    out: dict[str, Any] = {
        "data": page[: plan.limit],
        "limit": plan.limit,
        "offset": plan.offset,
        "has_next_page": len(page) > plan.limit,
    }
    if query_type == "raw":
        from .execute import render_sql
        out["raw_sql"] = render_sql(plan)
    return out


def _specs(plan: Plan, registry: dict | str) -> dict[str, MetricSpec]:
    if not plan.metrics:
        return {}
    if isinstance(registry, dict):
        declared = registry.get(plan.model.key) or {}
        return {name: MetricSpec(**declared[name]) if name in declared
                else MetricSpec(agg="count", approximate=True)
                for name in plan.metrics}
    from .metrics import resolve
    return {name: resolve(plan.model, name, registry) for name in plan.metrics}


def _spec_columns(spec: MetricSpec) -> set[str]:
    from .metrics import referenced_columns
    return referenced_columns(spec)


def _aggregate_periods(panel: list[dict], plan: Plan,
                       specs: dict[str, MetricSpec]) -> list[dict]:
    """Свернуть панель в динамику: группа = период × ключи группировки."""
    groups: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    for idx, row in enumerate(panel):
        key = (row["report_date"], *(row.get(c) for c in plan.columns))
        bucket = groups.get(key)
        if bucket is None:
            bucket = groups[key] = []
            order.append(key)
        bucket.append(idx)

    # Провайдер поверх панели: метрика считается по строкам периода, а не по
    # исходной витрине — иначе carry-forward не учтётся.
    def provider(name: str) -> list[Any]:
        return [row.get(name) for row in panel]

    out = []
    for key in order:
        row = {"report_date": key[0]}
        row.update(dict(zip(plan.columns, key[1:])))
        for name, spec in specs.items():
            row[name] = compute(spec, groups[key], provider)
        out.append(row)
    return out


def _order(rows: list[dict], plan: Plan) -> list[dict]:
    """Сортировка: явная, иначе по возрастанию периода.

    Динамика, отданная в произвольном порядке, читается как случайный набор
    чисел — поэтому период по умолчанию идёт по возрастанию.
    """
    if not plan.order_by:
        return sorted(rows, key=lambda r: (r.get("report_date") or "",
                                           str(r.get(ENTITY_KEY) or "")))
    from .execute import _sort
    return _sort(rows, plan)
