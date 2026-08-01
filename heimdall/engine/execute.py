"""Исполнение разобранного плана поверх колоночного снимка.

Конверт ответа — дословно из спеки: ``{data, limit, offset, has_next_page}``
плюс ``raw_sql`` только при ``query_type=raw``. Никаких метаданных о колонках:
агент видит ровно те ключи, которые запросил, и в этом весь смысл — по ответу
нельзя догадаться о существовании колонки, которую не запрашивал.

Порядок работы: план → маска фильтра → ветка режима → сортировка → limit_by →
срез страницы. Колонки читаются лениво: загружается только то, что упомянуто в
``columns``, ``filters``, ``order_by``, ``limit_by`` и в определениях метрик.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Protocol

from ..catalog.model import Model
from .compile import MODE_AGGREGATE, MODE_HISTORY, MODE_ROWS, Plan, compile_query
from .errors import fail
from .filters import collect_columns, evaluate
from .metrics import DEFAULT_REGISTRY, MetricSpec, compute, referenced_columns, resolve
from .quirks import EMBEDDING_DIMENSION_MISMATCH, Quirks


class Reader(Protocol):
    """Минимальный контракт источника данных."""

    rows: int

    def column(self, name: str) -> list[Any]: ...


def execute(body: dict, model: Model, reader: Reader, quirks: Quirks | None = None,
            *, query_type: str | None = None,
            metrics_registry: dict | str = DEFAULT_REGISTRY) -> dict:
    """Исполнить тело ``mcp_query`` и вернуть конверт ответа."""
    quirks = quirks or Quirks()
    plan = compile_query(body, model, quirks)
    provider = _lazy_provider(reader)

    if plan.mode == MODE_HISTORY:
        from .history import execute_history
        return execute_history(plan, reader, quirks, query_type=query_type,
                               metrics_registry=metrics_registry)

    mask = evaluate(plan.filters, model, reader.rows, provider, quirks)
    kept = [i for i, ok in enumerate(mask) if ok]

    specs = _metric_specs(plan, metrics_registry)
    _prefetch(plan, provider, specs)

    if plan.mode == MODE_AGGREGATE:
        rows = _aggregate(plan, kept, provider, specs)
    else:
        rows = _rows(plan, kept, provider, quirks)

    rows = _sort(rows, plan)
    if plan.limit_by:
        rows = _limit_by(rows, plan.limit_by)

    page = rows[plan.offset: plan.offset + plan.limit + 1]
    has_next = len(page) > plan.limit
    out: dict[str, Any] = {
        "data": page[: plan.limit],
        "limit": plan.limit,
        "offset": plan.offset,
        "has_next_page": has_next,
    }
    if query_type == "raw":
        out["raw_sql"] = render_sql(plan)
    return out


# ------------------------------------------------------------------ провайдер

def _lazy_provider(reader: Reader) -> Callable[[str], list[Any]]:
    cache: dict[str, list[Any]] = {}

    def get(name: str) -> list[Any]:
        values = cache.get(name)
        if values is None:
            values = cache[name] = reader.column(name)
        return values

    return get


def _prefetch(plan: Plan, provider: Callable[[str], list[Any]],
              specs: dict[str, MetricSpec]) -> None:
    """Тронуть все нужные колонки один раз — чтобы кэш читателя сработал."""
    wanted = set(plan.columns) | collect_columns(plan.filters)
    wanted |= {o["field"] for o in plan.order_by}
    if plan.limit_by:
        wanted |= set(plan.limit_by["by"])
    for spec in specs.values():
        wanted |= referenced_columns(spec)
    for name in sorted(wanted):
        provider(name)


def _metric_specs(plan: Plan, registry: dict | str) -> dict[str, MetricSpec]:
    """Разрешить все запрошенные метрики, включая параметрические."""
    out: dict[str, MetricSpec] = {}
    if isinstance(registry, dict):
        declared = registry.get(plan.model.key) or {}
        for name in plan.metrics + [p.name for p in plan.param_metrics]:
            spec = declared.get(name)
            out[name] = (MetricSpec(**spec) if spec
                         else MetricSpec(agg="count", approximate=True))
        return out
    for name in plan.metrics + [p.name for p in plan.param_metrics]:
        out[name] = resolve(plan.model, name, registry)
    return out


# --------------------------------------------------------------- режим строк

def _rows(plan: Plan, kept: list[int], provider: Callable[[str], list[Any]],
          quirks: Quirks) -> list[dict]:
    cols = {name: provider(name) for name in plan.columns}
    distances = _distances(plan, kept, provider, quirks)
    out = []
    for i in kept:
        row = {name: values[i] for name, values in cols.items()}
        for pname, values in distances.items():
            row[pname] = values[i]
        out.append(row)
    return out


def _distances(plan: Plan, kept: list[int], provider: Callable[[str], list[Any]],
               quirks: Quirks) -> dict[str, list[Any]]:
    """Посчитать параметрические колонки. Пока такая одна — косинусное расстояние."""
    out: dict[str, list[Any]] = {}
    for call in plan.param_columns:
        member = plan.model.columns[call.name]
        if member.name != "distance":
            # Прочие параметрические колонки берутся из снимка как есть.
            out[call.name] = provider(call.name)
            continue
        vector = call.args.get("query_vector") or []
        embeddings = provider("embedding")
        values: list[Any] = [None] * len(embeddings)
        for i in kept:
            emb = embeddings[i] or []
            if len(emb) != len(vector):
                if EMBEDDING_DIMENSION_MISMATCH in quirks:
                    # Каверза канала: без фильтра has_embedding = true
                    # cosineDistance применяется к пустому вектору и ClickHouse
                    # падает на несовпадении размерности при исполнении.
                    raise fail("internal-error",
                               f"cosineDistance: размерность вектора профиля {len(emb)} "
                               f"не совпадает с размерностью запроса {len(vector)}; "
                               f"добавь фильтр has_embedding = true")
                continue
            values[i] = _cosine_distance(emb, vector)
        out[call.name] = values
    return out


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 1.0
    return round(1.0 - dot / (na * nb), 6)


# ------------------------------------------------------------ режим агрегата

def _aggregate(plan: Plan, kept: list[int], provider: Callable[[str], list[Any]],
               specs: dict[str, MetricSpec]) -> list[dict]:
    keys = {name: provider(name) for name in plan.columns}
    groups: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    for i in kept:
        key = tuple(values[i] for values in keys.values())
        bucket = groups.get(key)
        if bucket is None:
            bucket = groups[key] = []
            order.append(key)
        bucket.append(i)
    if not plan.columns:
        # Общий итог: одна строка даже на пустой выборке — иначе агент решит,
        # что запрос не сработал, тогда как ответ «ноль» содержателен.
        order, groups = [()], {(): kept}

    args_by_metric = {p.name: p.args for p in plan.param_metrics}
    out = []
    for key in order:
        row = dict(zip(plan.columns, key))
        for name, spec in specs.items():
            row[name] = compute(spec, groups[key], provider, args_by_metric.get(name))
        out.append(row)
    return out


# ---------------------------------------------------------------- сортировка

_NULL = object()


def _sort(rows: list[dict], plan: Plan) -> list[dict]:
    """Устойчивая многоключевая сортировка с явной семантикой NULL.

    По умолчанию NULL идут первыми — это каверза канала: сортировка «по силе»
    без ``nulls: last`` поднимает наверх нескоренных.
    """
    for item in reversed(plan.order_by):
        field = item["field"]
        desc = item.get("direction", "asc") == "desc"
        nulls_last = item.get("nulls") == "last"

        def key(row: dict, _f=field, _nl=nulls_last, _d=desc):
            value = row.get(_f)
            if value is None:
                # Ключ пустоты выбирается так, чтобы после возможного разворота
                # NULL оказались там, где просили, а не там, где вышло.
                return (1 if _nl != _d else 0, _NullSentinel())
            return (0 if _nl != _d else 1, _Sortable(value))

        rows = sorted(rows, key=key, reverse=desc)
    return rows


class _NullSentinel:
    """Пустота сравнима сама с собой и ни с чем больше."""

    def __lt__(self, other) -> bool:
        return False

    def __gt__(self, other) -> bool:
        return False

    def __eq__(self, other) -> bool:
        return isinstance(other, _NullSentinel)

    def __hash__(self) -> int:
        return 0


class _Sortable:
    """Обёртка, сравнивающая разнородные значения без падения.

    ClickHouse не роняет запрос на смешанных типах в одной колонке; эмулятор
    тоже не должен — иначе тест упадёт там, где сервис ответит.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: "_Sortable") -> bool:
        a, b = self.value, other.value
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)

    def __eq__(self, other) -> bool:
        return isinstance(other, _Sortable) and self.value == other.value

    def __hash__(self) -> int:
        return hash(str(self.value))


def _limit_by(rows: list[dict], spec: dict) -> list[dict]:
    """``LIMIT N BY cols`` — N строк на группу, порядок сохраняется."""
    seen: dict[tuple, int] = {}
    out = []
    for row in rows:
        key = tuple(row.get(c) for c in spec["by"])
        count = seen.get(key, 0)
        if count < spec["limit"]:
            seen[key] = count + 1
            out.append(row)
    return out


# ------------------------------------------------------------------ raw_sql

def render_sql(plan: Plan) -> str:
    """Собрать SQL, эквивалентный плану.

    Это не то, что исполняется — исполняется план. Но агент использует
    ``raw_sql`` для самопроверки, и расхождение между показанным и посчитанным
    было бы худшим видом лжи, поэтому текст собирается из того же плана.
    """
    select: list[str] = list(plan.columns)
    for call in plan.param_columns:
        args = ", ".join(f"{k} = {_lit(v)}" for k, v in sorted(call.args.items()))
        select.append(f"{call.name}({args})")
    select += list(plan.metrics)
    for call in plan.param_metrics:
        args = ", ".join(f"{k} = {_lit(v)}" for k, v in sorted(call.args.items()))
        select.append(f"{call.name}({args})")

    parts = [f"SELECT {', '.join(select) or '*'}",
             f"FROM {plan.model.schema}.{plan.model.logic_model}"]
    if plan.filters:
        parts.append(f"WHERE {_sql_filter(plan.filters)}")
    if plan.mode == MODE_AGGREGATE and plan.columns:
        parts.append(f"GROUP BY {', '.join(plan.columns)}")
    if plan.order_by:
        items = []
        for o in plan.order_by:
            item = f"{o['field']} {o.get('direction', 'asc').upper()}"
            if o.get("nulls"):
                item += f" NULLS {o['nulls'].upper()}"
            items.append(item)
        parts.append(f"ORDER BY {', '.join(items)}")
    if plan.limit_by:
        parts.append(f"LIMIT {plan.limit_by['limit']} BY {', '.join(plan.limit_by['by'])}")
    parts.append(f"LIMIT {plan.limit}")
    if plan.offset:
        parts.append(f"OFFSET {plan.offset}")
    return "\n".join(parts)


def _sql_filter(node: dict) -> str:
    ntype = node["type"]
    if ntype in ("and", "or"):
        joiner = f" {ntype.upper()} "
        return "(" + joiner.join(_sql_filter(c) for c in node["conditions"]) + ")"
    if ntype == "not":
        return f"NOT ({_sql_filter(node['condition'])})"
    if ntype == "condition_null":
        return f"{node['column']} {node['operator']}"
    if ntype == "condition_like":
        return f"{node['column']} {node['operator']} {_lit(node['pattern'])}"
    if ntype == "condition_array":
        return f"{node['operator']}({node['column']}, {_lit(node['value'])})"
    if ntype == "condition_param":
        args = ", ".join(f"{k} = {_lit(v)}" for k, v in sorted((node.get("args") or {}).items()))
        return f"{node['name']}({args}) {node['operator']} {_lit(node['value'])}"
    return f"{node['column']} {node['operator']} {_lit(node['value'])}"


def _lit(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return f"[…{len(value)} значений…]"
        return "[" + ", ".join(_lit(v) for v in value) + "]"
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"
