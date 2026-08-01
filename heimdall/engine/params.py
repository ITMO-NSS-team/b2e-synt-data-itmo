"""Аргументы параметрических членов каталога.

Параметрический член — колонка или метрика, у которой в ``describe_model`` есть
поле ``parameters``. Голым именем её звать нельзя: ``columns: ["distance"]``
даёт ``param-virtual-not-supported``. Правильная форма — ``param_columns``
(или ``param_metrics``) со списком ``{name, args}``.

Проверки здесь — зеркало серверных кодов ``missing-parameter``,
``unknown-parameter``, ``parameter-type-invalid``, ``parameter-arity-mismatch``,
``parameter-too-many-values``, ``parameter-not-literal``.
"""
from __future__ import annotations

import re
from typing import Any

from ..catalog.model import Member
from .errors import fail

#: Потолок на длину списка-аргумента (вектор запроса, список id).
MAX_ARG_VALUES = 4096

_INT = re.compile(r"^U?Int\d+$", re.I)
_FLOAT = re.compile(r"^(Float\d+|Decimal.*|DECIMAL.*)$", re.I)
_DATE = re.compile(r"^Date\d*$", re.I)
_DATE_LITERAL = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_args(member: Member, args: dict[str, Any]) -> dict[str, Any]:
    """Проверить ``args`` вызова параметрического члена и вернуть их же.

    Если каталог не знает параметров члена (спека OpenAPI их не отдаёт, а
    курируемое дополнение ещё не заполнено) — проверяется только то, что
    аргументы вообще литералы. Придумывать несуществующие требования нельзя:
    ложный отказ хуже пропуска.
    """
    if not isinstance(args, dict):
        raise fail("request-validation-error", f"{member.name}: args должен быть объектом")

    for key, value in args.items():
        _assert_literal(member.name, key, value)

    declared = {p["name"]: p for p in member.parameters or []}
    if not declared:
        return args

    for name in declared:
        if name not in args:
            raise fail("missing-parameter",
                       f"{member.name}: не передан параметр {name} "
                       f"({declared[name].get('type', '?')})")
    for name in args:
        if name not in declared:
            raise fail("unknown-parameter",
                       f"{member.name}: параметра {name} нет; ожидаются "
                       f"{sorted(declared)}")
    for name, value in args.items():
        _check_type(member.name, name, declared[name].get("type", ""), value)
    return args


def _assert_literal(member: str, key: str, value: Any) -> None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return
    if isinstance(value, list):
        if len(value) > MAX_ARG_VALUES:
            raise fail("parameter-too-many-values",
                       f"{member}.{key}: {len(value)} значений при потолке {MAX_ARG_VALUES}")
        for v in value:
            if not isinstance(v, (str, int, float, bool)) and v is not None:
                raise fail("parameter-not-literal",
                           f"{member}.{key}: элементы аргумента должны быть литералами")
        return
    raise fail("parameter-not-literal",
               f"{member}.{key}: аргумент должен быть литералом или списком литералов")


def _check_type(member: str, name: str, declared: str, value: Any) -> None:
    decl = (declared or "").strip()
    array = decl.lower().startswith("array")
    if array:
        if not isinstance(value, list):
            raise fail("parameter-arity-mismatch",
                       f"{member}.{name}: объявлен {decl}, ожидается список")
        inner = decl[decl.find("(") + 1: decl.rfind(")")] if "(" in decl else ""
        for v in value:
            _check_scalar(member, name, inner, v)
        return
    if isinstance(value, list):
        raise fail("parameter-arity-mismatch",
                   f"{member}.{name}: объявлен {decl}, а передан список")
    _check_scalar(member, name, decl, value)


def _check_scalar(member: str, name: str, decl: str, value: Any) -> None:
    d = (decl or "").strip()
    if not d:
        return
    if _INT.match(d):
        if isinstance(value, bool) or not isinstance(value, int):
            raise fail("parameter-type-invalid",
                       f"{member}.{name}: объявлен {d}, а значение {value!r} не целое")
        return
    if _FLOAT.match(d) or d.lower() == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise fail("parameter-type-invalid",
                       f"{member}.{name}: объявлен {d}, а значение {value!r} не число")
        return
    if _DATE.match(d):
        if not (isinstance(value, str) and _DATE_LITERAL.match(value)):
            raise fail("parameter-type-invalid",
                       f"{member}.{name}: объявлен {d}, ожидается строка YYYY-MM-DD")
        return
    if d.lower().startswith("string") or d.lower() == "uuid":
        if not isinstance(value, str):
            raise fail("parameter-type-invalid",
                       f"{member}.{name}: объявлен {d}, а значение {value!r} не строка")
        return
    if d.lower() in ("bool", "boolean"):
        if not isinstance(value, bool):
            raise fail("parameter-type-invalid",
                       f"{member}.{name}: объявлен {d}, а значение {value!r} не булево")
