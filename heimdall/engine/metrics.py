"""Реализация метрик каталога.

Почему определения курируются, а не выводятся
---------------------------------------------
Спека перечисляет все 148 метрик по именам и даёт им русские описания, но не
даёт определений: тип у каждой указан как ``(String)`` — заглушка генератора
спеки, — а агрегата нет вовсе. Вывод по имени работает, но врёт на важном:
``cnt_vacancy`` выглядит как счётчик строк, а на деле это сумма коэффициентов
ставки (колонка ``vacancy`` имеет тип ``Float32``). Ровно такие расхождения
дают «200 OK и неверный ответ».

Поэтому: витрины, на которых работают скиллы, описаны явно в
``catalog/metrics.yaml``; остальное выводится по суффиксу имени и помечается
``approximate=True``. Флаг доходит до линтера, и правило, опирающееся на
приблизительную метрику, не может блокировать публикацию.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from ..catalog.model import Model
from .errors import fail

AGGS = frozenset({"count", "count_distinct", "sum", "min", "max", "avg", "ratio"})

#: Вывод агрегата по имени, когда определения нет. Порядок важен: первое
#: совпадение выигрывает.
_SUFFIX_AGG: tuple[tuple[str, str], ...] = (
    (r"_avg$|^avg_|_mean$", "avg"),
    (r"_sum$", "sum"),
    (r"^earliest_|_min$", "min"),
    (r"^latest_|_max$", "max"),
    (r"_count$|^cnt_|^count_|_qty$|_rows$|^headcount$|_cnt$", "count"),
)

DEFAULT_REGISTRY = "catalog/metrics.yaml"


@dataclass
class MetricSpec:
    """Определение одной метрики."""

    agg: str
    column: str | None = None
    where: dict | None = None
    numerator: dict | None = None
    denominator: dict | None = None
    #: True — определение выведено по имени, а не задано. Опираться можно, но
    #: блокировать по нему нельзя.
    approximate: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.agg not in AGGS:
            raise ValueError(f"неизвестный агрегат {self.agg!r}, ожидается один из {sorted(AGGS)}")


@lru_cache(maxsize=4)
def registry(path: str = DEFAULT_REGISTRY) -> dict[str, dict[str, dict]]:
    """Курируемые определения метрик, ключ — ``schema.model``."""
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("models") or {}


def resolve(model: Model, name: str, registry_path: str = DEFAULT_REGISTRY) -> MetricSpec:
    """Найти определение метрики модели или вывести его по имени."""
    if name not in model.metrics:
        raise fail("unknown-metric", f"в модели {model.key} нет метрики {name}")
    declared = (registry(registry_path).get(model.key) or {}).get(name)
    if declared:
        return MetricSpec(**{**declared, "approximate": False})
    return _guess(model, name)


def _guess(model: Model, name: str) -> MetricSpec:
    for pattern, agg in _SUFFIX_AGG:
        if re.search(pattern, name):
            if agg == "count":
                return MetricSpec(agg="count", approximate=True)
            column = _stem_column(model, name)
            # Без колонки-аргумента avg/sum/min/max посчитать не по чему —
            # честнее выродиться в count, чем притвориться средним.
            return (MetricSpec(agg=agg, column=column, approximate=True) if column
                    else MetricSpec(agg="count", approximate=True))
    return MetricSpec(agg="count", approximate=True)


def _stem_column(model: Model, name: str) -> str | None:
    stem = re.sub(r"^(earliest|latest|avg)_|_(avg|mean|sum|min|max)$", "", name)
    return stem if stem in model.columns else None


# ------------------------------------------------------------------- расчёт

def compute(spec: MetricSpec, rows: list[int], provider: Callable[[str], list[Any]],
            args: dict[str, Any] | None = None) -> Any:
    """Посчитать метрику по списку индексов строк.

    ``args`` — аргументы вызова параметрической метрики; они подставляются в
    предикат вида ``in_range: [date_from, date_to]``.
    """
    if spec.agg == "ratio":
        num = compute(MetricSpec(**spec.numerator), rows, provider, args)
        den = compute(MetricSpec(**spec.denominator), rows, provider, args)
        if not den:
            return None
        return round(num / den, 6)

    selected = _apply_where(spec.where, rows, provider, args or {})

    if spec.agg == "count":
        return len(selected)

    if spec.column is None:
        raise fail("unknown-metric",
                   f"агрегат {spec.agg} требует колонку-аргумент, а она не задана")
    values = provider(spec.column)

    if spec.agg == "count_distinct":
        return len({values[i] for i in selected if values[i] is not None})

    live = [values[i] for i in selected if values[i] is not None]
    if not live:
        # Ноль и «нет данных» — разные ответы. Смешивать их значит врать агенту:
        # средняя оценка 0 и отсутствие оценок читаются по-разному.
        return None
    if spec.agg == "sum":
        return _round(sum(live))
    if spec.agg == "min":
        return min(live)
    if spec.agg == "max":
        return max(live)
    return _round(sum(live) / len(live))


def _round(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def _apply_where(where: dict | None, rows: list[int],
                 provider: Callable[[str], list[Any]], args: dict[str, Any]) -> list[int]:
    if not where:
        return rows
    values = provider(where["column"])

    if "equals" in where:
        target = where["equals"]
        return [i for i in rows if values[i] == target]
    if "in" in where:
        allowed = set(where["in"])
        return [i for i in rows if values[i] in allowed]
    if where.get("not_null"):
        return [i for i in rows if values[i] is not None]
    if where.get("is_null"):
        return [i for i in rows if values[i] is None]
    if "in_range" in where:
        from_key, to_key = where["in_range"]
        lo, hi = args.get(from_key), args.get(to_key)
        if lo is None or hi is None:
            raise fail("missing-parameter",
                       f"метрика требует параметры {from_key} и {to_key}, "
                       f"передано {sorted(args)}")
        # Окно [from, to) — правая граница исключается, как у time_dimensions.
        return [i for i in rows if values[i] is not None and lo <= values[i] < hi]
    raise fail("request-validation-error", f"непонятный предикат метрики: {sorted(where)}")


def referenced_columns(spec: MetricSpec) -> set[str]:
    """Колонки, нужные для расчёта — чтобы грузить только их."""
    out: set[str] = set()
    if spec.column:
        out.add(spec.column)
    if spec.where:
        out.add(spec.where["column"])
    for side in (spec.numerator, spec.denominator):
        if side:
            out |= referenced_columns(MetricSpec(**side))
    return out
