"""Приведение литералов фильтра к типу колонки.

Здесь живут две вещи, которые легко перепутать:

* **валидация** — «строка вместо числа» должна давать ``filter-value-invalid``;
* **каверза** — строка ``'false'`` на ``Bool``-колонке должна тихо стать ``True``
  и инвертировать выборку, потому что именно так ведёт себя канал.

Разница принципиальная: первое агент видит и чинит, второе — не видит.
"""
from __future__ import annotations

import datetime as dt
import re
import uuid
from typing import Any

from ..catalog.types import (ChType, KIND_BOOL, KIND_DATE, KIND_DATETIME, KIND_DECIMAL,
                             KIND_FLOAT, KIND_INT, KIND_STRING, KIND_UUID)
from .errors import fail
from .quirks import BOOL_STRING_COERCION, Quirks

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def coerce(value: Any, ch: ChType, quirks: Quirks, *, column: str) -> Any:
    """Привести значение литерала к типу колонки или упасть с filter-value-invalid."""
    if value is None:
        return None

    kind = ch.kind
    if kind == KIND_BOOL:
        return _coerce_bool(value, quirks, column)
    if kind == KIND_INT:
        return _coerce_int(value, column)
    if kind in (KIND_FLOAT, KIND_DECIMAL):
        return _coerce_float(value, column)
    if kind == KIND_DATE:
        return _coerce_date(value, column)
    if kind == KIND_DATETIME:
        return _coerce_datetime(value, column)
    if kind == KIND_UUID:
        return _coerce_uuid(value, column)
    if kind == KIND_STRING:
        if isinstance(value, bool):
            raise fail("filter-value-invalid",
                       f"колонка {column} строковая, а значение булево")
        return str(value)
    # Тип элемента массива спека не сообщает — сравниваем как есть.
    return value


def _coerce_bool(value: Any, quirks: Quirks, column: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if BOOL_STRING_COERCION in quirks:
            # Каверза канала: непустая строка коэрсится в true. 'false' → True,
            # и выборка инвертируется молча.
            return bool(value)
        if value.lower() in ("true", "1"):
            return True
        if value.lower() in ("false", "0"):
            return False
        raise fail("filter-value-invalid",
                   f"колонка {column} булева, а значение {value!r} не булево")
    if isinstance(value, int):
        return bool(value)
    raise fail("filter-value-invalid", f"колонка {column} булева, а значение {value!r} не булево")


def _coerce_int(value: Any, column: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    raise fail("filter-value-invalid", f"колонка {column} целочисленная, а значение {value!r} — нет")


def _coerce_float(value: Any, column: str) -> float:
    if isinstance(value, bool):
        raise fail("filter-value-invalid", f"колонка {column} числовая, а значение булево")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise fail("filter-value-invalid", f"колонка {column} числовая, а значение {value!r} — нет")


def _coerce_date(value: Any, column: str) -> str:
    if isinstance(value, str) and _DATE.match(value):
        try:
            dt.date.fromisoformat(value)
        except ValueError:
            raise fail("filter-value-invalid", f"{column}: {value!r} не существующая дата")
        return value
    raise fail("filter-value-invalid",
               f"колонка {column} — дата, ожидается строка YYYY-MM-DD, пришло {value!r}")


def _coerce_datetime(value: Any, column: str) -> str:
    if isinstance(value, str):
        head = value.replace("T", " ")[:19]
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt.datetime.strptime(head, fmt)
                return value
            except ValueError:
                continue
    raise fail("filter-value-invalid",
               f"колонка {column} — время, ожидается YYYY-MM-DD[ HH:MM:SS], пришло {value!r}")


def _coerce_uuid(value: Any, column: str) -> str:
    if isinstance(value, str):
        try:
            uuid.UUID(value)
            return value
        except ValueError:
            pass
    raise fail("filter-value-invalid", f"колонка {column} — UUID, а значение {value!r} не UUID")


def compare(left: Any, op: str, right: Any) -> bool:
    """Сравнение с семантикой NULL: любое сравнение с NULL — ложь."""
    if left is None or right is None:
        return False
    try:
        if op == "=":
            return left == right
        if op == "!=":
            return left != right
        if op == ">":
            return left > right
        if op == "<":
            return left < right
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
    except TypeError:
        return False
    raise fail("request-validation-error", f"неизвестный оператор сравнения {op!r}")
