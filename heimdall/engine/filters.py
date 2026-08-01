"""Дерево фильтров ``mcp_query``: восемь типов узлов, ровно один корень.

Грамматика взята из спеки дословно, включая ``condition_param`` — узел, который
есть в API и в тексте ``get_overview``, но отсутствует в приёме ``filters``.
Именно поэтому эмулятор обязан его поддерживать: иначе фабрика никогда не
узнает, что агенты о нём не знают.

Каждый узел проверяется на лишние поля: у настоящего сервиса
``additionalProperties: false``, и самый частый реальный дефект — значение
``condition_like`` в ``value`` вместо ``pattern`` — отвергает всё тело целиком.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Callable

from ..catalog.model import Model
from ..catalog.types import KIND_STRING, KIND_TUPLE
from .errors import fail
from .quirks import Quirks, fold_case
from .values import coerce, compare

LOGICAL = {"and", "or", "not"}

# Поля узлов взяты из per-model схем спеки (_EqFilter, _InFilter, _LikeFilter,
# _NullFilter, _ArrayFilter, ParamCondFilter). Два поля не описаны ни в CLAUDE.md,
# ни в приёмах, но объявлены в спеке и потому обязаны работать:
#   is_tuple — механизм составного ключа (emp_key), default false;
#   cast     — приведение типа перед сравнением, default "".
_NODE_FIELDS: dict[str, set[str]] = {
    "condition": {"type", "column", "operator", "value", "is_tuple", "cast"},
    "condition_in": {"type", "column", "operator", "value", "is_tuple"},
    "condition_like": {"type", "column", "operator", "pattern", "case_sensitive"},
    "condition_null": {"type", "column", "operator"},
    "condition_array": {"type", "column", "operator", "value", "flatten", "cast"},
    "condition_param": {"type", "name", "args", "operator", "value", "expr"},
    "and": {"type", "conditions"},
    "or": {"type", "conditions"},
    "not": {"type", "condition"},
}

_REQUIRED: dict[str, set[str]] = {
    "condition": {"column", "operator", "value"},
    "condition_in": {"column", "operator", "value"},
    "condition_like": {"column", "operator", "pattern"},
    "condition_null": {"column", "operator"},
    "condition_array": {"column", "operator", "value"},
    "condition_param": {"name", "operator", "value"},
    "and": {"conditions"},
    "or": {"conditions"},
    "not": {"condition"},
}

_OPERATORS: dict[str, set[str]] = {
    "condition": {"=", "!=", ">", "<", ">=", "<="},
    "condition_in": {"IN", "NOT IN"},
    "condition_like": {"LIKE", "NOT LIKE", "ILIKE", "NOT ILIKE"},
    "condition_null": {"IS NULL", "IS NOT NULL"},
    "condition_array": {"has", "hasAny", "hasAll"},
    "condition_param": {"=", "!=", ">", "<", ">=", "<="},
}

#: Провайдер значений колонки: имя → список значений по строкам.
ColumnProvider = Callable[[str], list[Any]]


def validate(node: Any, model: Model) -> None:
    """Статически проверить дерево фильтров — без данных.

    Отдельная от ``evaluate`` функция нужна линтеру: он обязан ловить дефект
    грамматики до исполнения и без снимка данных. Проверки при этом ровно те
    же, что применит сервис, потому что применяет их этот же код.
    """
    if node is None:
        return
    _check_shape(node)
    ntype = node["type"]

    if ntype in ("and", "or"):
        conditions = node["conditions"]
        if not isinstance(conditions, list) or not conditions:
            raise fail("request-validation-error",
                       f"узел {ntype}: conditions — непустой список")
        for child in conditions:
            validate(child, model)
        return
    if ntype == "not":
        validate(node["condition"], model)
        return

    op = node.get("operator")
    if op not in _OPERATORS[ntype]:
        raise fail("request-validation-error",
                   f"узел {ntype}: оператор {op!r} недопустим, ожидается один из "
                   f"{sorted(_OPERATORS[ntype])}")

    name = node.get("name") if ntype == "condition_param" else node.get("column")
    member = model.columns.get(name)
    if member is None:
        if ntype == "condition_param" and name in model.metrics:
            raise fail("param-metric-not-supported",
                       f"{name} — параметрическая метрика, в filters её нельзя (агрегат)")
        raise fail("unknown-column", f"в модели {model.key} нет колонки {name}")

    if ntype == "condition_param":
        if member.kind != "param_column":
            raise fail("request-validation-error",
                       f"{name} — не параметрическая колонка, адресуй её полем column")
        return
    if member.kind == "param_column":
        raise fail("param-virtual-not-supported",
                   f"{name} — параметрическая колонка, в фильтре адресуй её узлом "
                   f"condition_param полем name")

    ch = member.ch
    if ntype == "condition_array" and not ch.is_array:
        raise fail("array-operator-requires-array-column",
                   f"{name} не Array, оператор {op} к ней неприменим")
    if ntype != "condition_array" and ntype != "condition_null" and ch.is_array:
        raise fail("request-validation-error",
                   f"{name} — Array-колонка, фильтруй её узлом condition_array")
    if ntype == "condition_like" and not ch.is_filterable_by_like:
        raise fail("request-validation-error",
                   f"condition_like работает только по строковой скалярной колонке, "
                   f"{name} имеет тип {member.type}")
    if ntype == "condition_in" and not isinstance(node.get("value"), list):
        raise fail("filter-value-invalid", f"{name}: оператор {op} требует список")


def _check_shape(node: Any) -> None:
    """Форма узла: тип, отсутствие лишних полей, наличие обязательных."""
    if not isinstance(node, dict):
        raise fail("request-validation-error", "узел фильтра должен быть объектом")

    ntype = node.get("type")
    if not ntype:
        raise fail("request-validation-error", "у каждого узла фильтра обязано быть поле type")
    if ntype not in _NODE_FIELDS:
        raise fail("request-validation-error", f"неизвестный тип узла фильтра: {ntype!r}")

    extra = set(node) - _NODE_FIELDS[ntype]
    if extra:
        # Дословно как у сервиса: лишнее поле отвергает всё тело. Это ловит
        # самый частый дефект живого корпуса — значение like в `value`.
        raise fail("request-validation-error",
                   f"узел {ntype}: лишние поля {sorted(extra)}",
                   errors=[{"loc": ["filters", ntype, f], "msg": "extra fields not permitted",
                            "type": "value_error.extra"} for f in sorted(extra)])
    missing = _REQUIRED[ntype] - set(node)
    if missing:
        raise fail("request-validation-error",
                   f"узел {ntype}: не хватает полей {sorted(missing)}")


def evaluate(node: dict, model: Model, rows: int, provider: ColumnProvider,
             quirks: Quirks) -> list[bool]:
    """Свести дерево фильтров к булевой маске длиной ``rows``."""
    if node is None:
        return [True] * rows
    _check_shape(node)
    ntype = node["type"]

    if ntype in ("and", "or"):
        parts = _children(node, model, rows, provider, quirks)
        if not parts:
            # minItems: 1 в спеке — пустой список условий это ошибка тела,
            # а не «всегда истина».
            raise fail("request-validation-error",
                       f"узел {ntype}: conditions не может быть пустым")
        merge = all if ntype == "and" else any
        return [merge(vals) for vals in zip(*parts)]
    if ntype == "not":
        inner = evaluate(node["condition"], model, rows, provider, quirks)
        return [not v for v in inner]

    op = node.get("operator")
    if op not in _OPERATORS[ntype]:
        raise fail("request-validation-error",
                   f"узел {ntype}: оператор {op!r} недопустим, ожидается один из "
                   f"{sorted(_OPERATORS[ntype])}")

    if ntype == "condition_param":
        return _eval_param(node, model, rows, provider, quirks)

    column = node["column"]
    member = model.columns.get(column)
    if member is None:
        raise fail("unknown-column", f"в модели {model.key} нет колонки {column}")
    if member.kind == "param_column":
        raise fail("param-virtual-not-supported",
                   f"{column} — параметрическая колонка, в фильтре адресуй её узлом "
                   f"condition_param полем name")

    values = provider(column)
    ch = member.ch

    if ntype == "condition_null":
        want_null = op == "IS NULL"
        return [(v is None) == want_null for v in values]

    if ntype == "condition_array":
        if not ch.is_array:
            raise fail("array-operator-requires-array-column",
                       f"{column} не Array, оператор {op} к ней неприменим")
        # Спека: «массив с листом Tuple/Map отклоняется — разворачивайте его
        # через ARRAY JOIN». Класс листа известен только из курируемого
        # дополнения catalog/array_elements.yaml.
        leaf = (member.element_type or "").strip().lower()
        if leaf.startswith(("tuple", "map")):
            raise fail("array-operator-requires-array-column",
                       f"{column} — Array({member.element_type}); сравнение со скаляром "
                       f"отклоняется, разворачивай через ARRAY JOIN или модель-развёртку")
        return _eval_array(node, values, op)

    if ch.is_array:
        raise fail("request-validation-error",
                   f"{column} — Array-колонка, фильтруй её узлом condition_array "
                   f"(has/hasAny/hasAll)")

    if ntype == "condition_like":
        if not ch.is_filterable_by_like:
            raise fail("request-validation-error",
                       f"condition_like работает только по строковой скалярной колонке, "
                       f"{column} имеет тип {member.type}")
        return _eval_like(node, values, op, quirks)

    if ch.kind == KIND_TUPLE:
        return _eval_tuple(node, values, op, column)

    if ntype == "condition_in":
        raw = node["value"]
        if not isinstance(raw, list) or not raw:
            raise fail("filter-value-invalid",
                       f"{column}: оператор {op} требует непустой список значений")
        wanted = [coerce(v, ch, quirks, column=column) for v in raw]
        inside = [v is not None and v in wanted for v in values]
        return inside if op == "IN" else [not x for x in inside]

    wanted = coerce(node["value"], ch, quirks, column=column)
    return [compare(v, op, wanted) for v in values]


def _children(node: dict, model: Model, rows: int, provider: ColumnProvider,
              quirks: Quirks) -> list[list[bool]]:
    conds = node["conditions"]
    if not isinstance(conds, list):
        raise fail("request-validation-error", f"узел {node['type']}: conditions должен быть списком")
    return [evaluate(c, model, rows, provider, quirks) for c in conds]


def _eval_like(node: dict, values: list[Any], op: str, quirks: Quirks) -> list[bool]:
    pattern = node["pattern"]
    if not isinstance(pattern, str):
        raise fail("filter-value-invalid", "pattern должен быть строкой")
    # По умолчанию регистр определяет оператор; явный case_sensitive его перебивает
    # (в спеке поле объявлено с default true и в приёмах помечено как REST-only).
    insensitive = op.endswith("ILIKE")
    explicit = node.get("case_sensitive")
    if explicit is not None:
        insensitive = not bool(explicit)
    glob = pattern.replace("%", "*").replace("_", "?")
    if insensitive:
        glob = fold_case(glob, quirks)

    out: list[bool] = []
    for v in values:
        if v is None:
            out.append(False)
            continue
        text = str(v)
        if insensitive:
            text = fold_case(text, quirks)
        out.append(fnmatch.fnmatchcase(text, glob))
    return out if not op.startswith("NOT") else [
        (not m) and v is not None for m, v in zip(out, values)]


def _eval_array(node: dict, values: list[Any], op: str) -> list[bool]:
    raw = node["value"]
    flatten = bool(node.get("flatten", False))

    def elems(cell: Any) -> list[Any]:
        if cell is None:
            return []
        if not isinstance(cell, (list, tuple)):
            return [cell]
        if flatten or any(isinstance(x, (list, tuple)) for x in cell):
            out: list[Any] = []
            stack = list(cell)
            while stack:
                x = stack.pop(0)
                if isinstance(x, (list, tuple)):
                    stack = list(x) + stack
                else:
                    out.append(x)
            return out
        return list(cell)

    if op == "has":
        if isinstance(raw, list):
            raise fail("filter-value-invalid", "оператор has принимает скаляр, а не список")
        return [raw in elems(c) for c in values]

    if not isinstance(raw, list) or not raw:
        raise fail("filter-value-invalid", f"оператор {op} требует непустой список значений")
    if op == "hasAny":
        return [bool(set(elems(c)) & set(raw)) for c in values]
    return [set(raw) <= set(elems(c)) for c in values]


def _eval_tuple(node: dict, values: list[Any], op: str, column: str) -> list[bool]:
    """Составной ключ: только =, !=, IN, NOT IN, с проверкой арности."""
    if op not in ("=", "!=", "IN", "NOT IN"):
        raise fail("composite-key-operator-unsupported",
                   f"{column} — Tuple, оператор {op} недопустим")

    def as_tuple(v: Any) -> tuple:
        if not isinstance(v, (list, tuple)):
            raise fail("filter-value-invalid",
                       f"{column}: значение кортежа передаётся массивом, пришло {v!r}")
        return tuple(v)

    arity = next((len(v) for v in values if isinstance(v, (list, tuple))), None)
    if op in ("=", "!="):
        wanted = {as_tuple(node["value"])}
    else:
        raw = node["value"]
        if not isinstance(raw, list) or not raw:
            raise fail("filter-value-invalid", f"{column}: {op} требует непустой список кортежей")
        wanted = {as_tuple(v) for v in raw}
    if arity is not None and any(len(w) != arity for w in wanted):
        raise fail("composite-key-arity-mismatch",
                   f"{column}: арность кортежа {arity}, а в фильтре другая")

    inside = [v is not None and tuple(v) in wanted for v in values]
    return inside if op in ("=", "IN") else [not x for x in inside]


def _eval_param(node: dict, model: Model, rows: int, provider: ColumnProvider,
                quirks: Quirks) -> list[bool]:
    """Предикат по параметрическому члену: адресуется полем name, не column."""
    name = node["name"]
    member = model.columns.get(name)
    if member is None or member.kind != "param_column":
        if name in model.metrics:
            raise fail("param-metric-not-supported",
                       f"{name} — параметрическая метрика, в filters её нельзя (агрегат)")
        raise fail("unknown-column", f"в модели {model.key} нет параметрической колонки {name}")

    from .params import check_args      # локальный импорт: параметры знают про каталог
    check_args(member, node.get("args") or {})

    values = provider(name)
    wanted = coerce(node["value"], member.ch, quirks, column=name)
    return [compare(v, node["operator"], wanted) for v in values]


def collect_columns(node: dict | None, out: set[str] | None = None) -> set[str]:
    """Все имена колонок, упомянутые в дереве — для ленивой загрузки."""
    out = out if out is not None else set()
    if not isinstance(node, dict):
        return out
    if "column" in node and isinstance(node["column"], str):
        out.add(node["column"])
    if node.get("type") == "condition_param" and isinstance(node.get("name"), str):
        out.add(node["name"])
    for child in node.get("conditions", []) or []:
        collect_columns(child, out)
    if node.get("condition"):
        collect_columns(node["condition"], out)
    return out
