"""Разбор тела ``mcp_query`` в план исполнения.

Главная механика канала: **режим не задаётся полем, он выводится из структуры
тела**. Это самый частый источник ошибок класса «сгенерировали синтаксически
валидное, но семантически другое», поэтому вывод режима вынесен в отдельную
функцию и покрыт тестами по таблице истинности.

::

    time_dimensions             → history
    metrics | param_metrics     → aggregate
    только columns              → rows
    ничего                      → query-empty
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from ..catalog.model import Model
from .errors import fail
from .params import check_args
from .quirks import SILENT_LIMIT_CLAMP, Quirks

#: Потолок строк на стороне сервиса. Значение выше ужимается МОЛЧА.
API_MAX_LIMIT = 1000
DEFAULT_LIMIT = 100

MODE_ROWS = "rows"
MODE_AGGREGATE = "aggregate"
MODE_HISTORY = "history"

BODY_FIELDS = {"schema", "logic_model", "columns", "metrics", "filters", "order_by",
               "time_dimensions", "limit", "offset", "limit_by", "param_metrics",
               "param_columns"}

GRANULARITY_STEP = {
    "day": "day",
    "week_end": "week",
    "month_end": "month",
    "quarter_end": "quarter",
    "year_end": "year",
}


@dataclass
class ParamCall:
    name: str
    args: dict[str, Any]

    @property
    def output_name(self) -> str:
        return self.name


@dataclass
class TimeWindow:
    name: str
    granularity: str
    date_from: str
    date_to: str
    fill: str = "previous"


@dataclass
class Plan:
    """Разобранное тело запроса: что именно и как считать."""

    model: Model
    mode: str
    columns: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    param_columns: list[ParamCall] = field(default_factory=list)
    param_metrics: list[ParamCall] = field(default_factory=list)
    filters: dict | None = None
    order_by: list[dict] = field(default_factory=list)
    limit: int = DEFAULT_LIMIT
    offset: int = 0
    limit_by: dict | None = None
    window: TimeWindow | None = None
    #: Запрошенный лимит до молчаливого ужатия — для трейса и чекеров.
    requested_limit: int = DEFAULT_LIMIT

    @property
    def output_fields(self) -> list[str]:
        out = list(self.columns)
        if self.window:
            out.insert(0, self.window.name)
        out += [p.output_name for p in self.param_columns]
        out += list(self.metrics)
        out += [p.output_name for p in self.param_metrics]
        return out


def infer_mode(body: dict) -> str:
    """Вывести режим из структуры тела. Отдельная функция — потому что это контракт."""
    if body.get("time_dimensions"):
        return MODE_HISTORY
    if body.get("metrics") or body.get("param_metrics"):
        return MODE_AGGREGATE
    if body.get("columns"):
        return MODE_ROWS
    raise fail("query-empty",
               "не заданы ни columns, ни metrics, ни time_dimensions — нечего возвращать")


def compile_query(body: dict, model: Model, quirks: Quirks | None = None) -> Plan:
    """Проверить тело и собрать план. Все отказы — доменными кодами, не исключениями Python."""
    quirks = quirks or Quirks()
    if not isinstance(body, dict):
        raise fail("request-validation-error", "тело запроса должно быть объектом")

    extra = set(body) - BODY_FIELDS
    if extra:
        raise fail("request-validation-error",
                   f"лишние поля в теле: {sorted(extra)}",
                   errors=[{"loc": ["body", f], "msg": "extra fields not permitted",
                            "type": "value_error.extra"} for f in sorted(extra)])

    mode = infer_mode(body)
    plan = Plan(model=model, mode=mode, filters=body.get("filters"))

    # Дерево фильтров проверяется статически, здесь же: иначе дефект грамматики
    # всплыл бы только на исполнении, а линтеру нужно поймать его без данных.
    from .filters import validate as validate_filters
    validate_filters(plan.filters, model)

    _resolve_columns(plan, body, mode)
    _resolve_metrics(plan, body, mode)
    _resolve_param_members(plan, body, mode)
    if mode == MODE_HISTORY:
        _resolve_window(plan, body)
    _resolve_order_by(plan, body, mode)
    _resolve_limits(plan, body, mode, quirks)
    return plan


# ------------------------------------------------------------------- разрешение

def _resolve_columns(plan: Plan, body: dict, mode: str) -> None:
    names = body.get("columns") or []
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise fail("request-validation-error",
                   "columns — список ИМЁН-СТРОК; объектная форма {name, args} только у param_columns")
    model = plan.model
    for name in names:
        member = model.columns.get(name)
        if member is None:
            if name in model.metrics:
                raise fail("unknown-column",
                           f"{name} — метрика модели {model.key}, её место в metrics, не в columns")
            raise fail("unknown-column", f"в модели {model.key} нет колонки {name}")
        if member.kind == "param_column":
            raise fail("param-virtual-not-supported",
                       f"{name} — параметрическая колонка, передавай её в param_columns "
                       f"как {{name, args}}")
        if mode == MODE_HISTORY and name in model.time_dimensions:
            raise fail("request-validation-error",
                       f"{name} — ось истории, она и так в выдаче; убери её из columns")
    plan.columns = list(names)


def _resolve_metrics(plan: Plan, body: dict, mode: str) -> None:
    names = body.get("metrics") or []
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise fail("request-validation-error",
                   "metrics — список ИМЁН-СТРОК; параметрические метрики идут в param_metrics")
    model = plan.model
    for name in names:
        member = model.metrics.get(name)
        if member is None:
            if name in model.columns:
                raise fail("unknown-metric",
                           f"{name} — колонка модели {model.key}, её место в columns, не в metrics")
            raise fail("unknown-metric", f"в модели {model.key} нет метрики {name}")
        if member.kind == "param_metric":
            raise fail("param-metric-not-supported",
                       f"{name} — параметрическая метрика, передавай её в param_metrics "
                       f"как {{name, args}}")
    plan.metrics = list(names)


def _resolve_param_members(plan: Plan, body: dict, mode: str) -> None:
    model = plan.model
    for field_name, kind, container in (("param_columns", "param_column", plan.param_columns),
                                        ("param_metrics", "param_metric", plan.param_metrics)):
        calls = body.get(field_name) or []
        if not isinstance(calls, list):
            raise fail("request-validation-error", f"{field_name} должен быть списком объектов")
        if calls and mode == MODE_HISTORY:
            code = ("param-virtual-not-supported-in-history" if kind == "param_column"
                    else "param-metric-not-supported-in-history")
            raise fail(code, f"{field_name} в режиме истории не поддержаны")
        for call in calls:
            if not isinstance(call, dict) or "name" not in call:
                raise fail("request-validation-error",
                           f"{field_name}: элемент должен быть объектом с полем name")
            unknown = set(call) - {"name", "args"}
            if unknown:
                raise fail("request-validation-error",
                           f"{field_name}: лишние поля {sorted(unknown)}")
            name = call["name"]
            pool = model.columns if kind == "param_column" else model.metrics
            member = pool.get(name)
            if member is None or member.kind != kind:
                if member is not None:
                    raise fail("request-validation-error",
                               f"{name} — не параметрический член, передавай его "
                               f"{'в columns' if kind == 'param_column' else 'в metrics'}")
                code = ("unknown-column" if kind == "param_column" else "unknown-metric")
                raise fail(code, f"в модели {model.key} нет параметрического члена {name}")
            args = check_args(member, call.get("args") or {})
            container.append(ParamCall(name=name, args=args))


def _resolve_window(plan: Plan, body: dict) -> None:
    model = plan.model
    dims = body["time_dimensions"]
    if not isinstance(dims, list) or len(dims) != 1:
        raise fail("request-validation-error",
                   "time_dimensions — ровно один элемент")
    if not model.is_history:
        raise fail("column-not-a-time-dimension",
                   f"модель {model.key} не историческая: режим history ей недоступен")
    dim = dims[0]
    unknown = set(dim) - {"name", "granularity", "range", "fill"}
    if unknown:
        raise fail("request-validation-error", f"time_dimensions: лишние поля {sorted(unknown)}")

    name = dim.get("name")
    td = model.time_dimensions.get(name)
    if td is None:
        raise fail("column-not-a-time-dimension",
                   f"{name} не объявлена time-dimension модели {model.key}; "
                   f"доступны {sorted(model.time_dimensions)}")
    gran = dim.get("granularity")
    if gran not in td.granularities:
        raise fail("granularity-not-allowed",
                   f"гранулярность {gran!r} недопустима; доступны {td.granularities}")

    rng = dim.get("range") or {}
    if rng.get("type", "absolute") != "absolute":
        raise fail("request-validation-error",
                   "range.type поддерживается только absolute: относительных окон нет")
    date_from, date_to = rng.get("from"), rng.get("to")
    for label, value in (("from", date_from), ("to", date_to)):
        if not isinstance(value, str):
            raise fail("request-validation-error", f"range.{label} обязателен, формат YYYY-MM-DD")
        try:
            dt.date.fromisoformat(value)
        except ValueError:
            raise fail("request-validation-error",
                       f"range.{label}={value!r}: ожидается YYYY-MM-DD")
    if date_to <= date_from:
        raise fail("request-validation-error",
                   "окно задаётся как [from, to), правая граница исключается: to должно быть больше from")

    span_days = (dt.date.fromisoformat(date_to) - dt.date.fromisoformat(date_from)).days
    if span_days > model.default_lookback_years * 366:
        raise fail("range-exceeds-lookback",
                   f"окно {span_days} дней шире глубины истории "
                   f"{model.default_lookback_years} лет")

    fill = (dim.get("fill") or {}).get("mode", "previous")
    if fill not in ("previous", "none"):
        raise fail("request-validation-error", "fill.mode ∈ {previous, none}")
    plan.window = TimeWindow(name=name, granularity=gran, date_from=date_from,
                             date_to=date_to, fill=fill)


def _resolve_order_by(plan: Plan, body: dict, mode: str) -> None:
    items = body.get("order_by") or []
    if not isinstance(items, list):
        raise fail("request-validation-error", "order_by должен быть списком")
    model = plan.model
    for item in items:
        if not isinstance(item, dict) or "field" not in item:
            raise fail("request-validation-error", "order_by: элемент должен иметь поле field")
        unknown = set(item) - {"field", "kind", "direction", "nulls"}
        if unknown:
            raise fail("request-validation-error", f"order_by: лишние поля {sorted(unknown)}")
        if item.get("direction", "asc") not in ("asc", "desc"):
            raise fail("request-validation-error", "order_by.direction ∈ {asc, desc}")
        if item.get("nulls") not in (None, "first", "last"):
            raise fail("request-validation-error", "order_by.nulls ∈ {first, last}")

        field_name = item["field"]
        is_metric = field_name in model.metrics or field_name in {p.name for p in plan.param_metrics}
        is_column = field_name in model.columns
        is_time = plan.window is not None and field_name == plan.window.name
        if not (is_metric or is_column or is_time):
            raise fail("unknown-column",
                       f"order_by.field={field_name} нет ни в колонках, ни в метриках {model.key}")

        declared = item.get("kind")
        actual = "metric" if is_metric else ("time_dimension" if is_time else "column")
        if declared and declared != actual:
            raise fail("order-by-kind-mismatch",
                       f"order_by.field={field_name} — {actual}, а заявлен как {declared}")
        if actual == "metric" and mode == MODE_ROWS:
            raise fail("order-by-metric-not-allowed-in-rows",
                       f"в режиме строк нельзя сортировать по метрике {field_name}: "
                       f"метрик в выдаче нет")
        if actual == "metric" and field_name not in plan.metrics and \
                field_name not in {p.name for p in plan.param_metrics}:
            raise fail("request-validation-error",
                       f"сортировка по метрике {field_name}, которой нет в metrics запроса")
        if actual == "column" and mode == MODE_AGGREGATE and field_name not in plan.columns:
            raise fail("request-validation-error",
                       f"в режиме агрегата сортировать можно только по ключам группировки; "
                       f"{field_name} не входит в columns")
    plan.order_by = list(items)


def _resolve_limits(plan: Plan, body: dict, mode: str, quirks: Quirks) -> None:
    limit = body.get("limit", DEFAULT_LIMIT)
    if limit is None:
        limit = DEFAULT_LIMIT
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise fail("request-validation-error", "limit — целое число ≥ 1")
    plan.requested_limit = limit
    if limit > API_MAX_LIMIT:
        if SILENT_LIMIT_CLAMP in quirks:
            # Каверза: ужимаем МОЛЧА. Применённое значение видно только в ответе.
            limit = API_MAX_LIMIT
        else:
            raise fail("request-validation-error",
                       f"limit {limit} выше потолка канала {API_MAX_LIMIT}")
    plan.limit = limit

    offset = body.get("offset", 0) or 0
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise fail("request-validation-error", "offset — целое число ≥ 0")
    plan.offset = offset

    lb = body.get("limit_by")
    if lb is None:
        return
    if mode == MODE_HISTORY:
        raise fail("limit-by-not-supported-in-history", "limit_by в режиме истории не поддержан")
    if not isinstance(lb, dict) or set(lb) - {"limit", "by"} or "limit" not in lb or "by" not in lb:
        raise fail("request-validation-error", "limit_by — объект {limit, by}")
    if not isinstance(lb["limit"], int) or lb["limit"] < 1:
        raise fail("request-validation-error", "limit_by.limit — целое ≥ 1")
    by = lb["by"]
    if not isinstance(by, list) or not by:
        raise fail("request-validation-error", "limit_by.by — непустой список колонок")
    for name in by:
        if name not in plan.model.columns:
            raise fail("unknown-column", f"limit_by.by: в модели {plan.model.key} нет колонки {name}")
    plan.limit_by = {"limit": lb["limit"], "by": list(by)}
