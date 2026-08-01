"""Разбор типов ClickHouse в том виде, в каком их отдаёт спека Heimdall.

Спека пишет тип строкой в скобках после имени колонки:

    - `company` (String (LowCardinality)): Компания
    - `employee_id_other` (Array (nullable)): Табельные номера совмещения
    - `absence_days` (Map(String, UInt8)): ...

Формат небогатый: у ``Array`` не указан тип элемента, у ``Tuple`` не указана
арность. Всё, что можно достать, достаётся здесь; всё, чего нельзя, помечается
как неизвестное и добирается курируемыми файлами (`catalog/array_elements.yaml`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Скалярные семейства, к которым сводится всё остальное. Нужны генератору
# данных (какое значение породить) и линтеру (какой литерал допустим в фильтре).
KIND_STRING = "string"
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_BOOL = "bool"
KIND_DATE = "date"
KIND_DATETIME = "datetime"
KIND_UUID = "uuid"
KIND_DECIMAL = "decimal"
KIND_MAP = "map"
KIND_TUPLE = "tuple"
KIND_UNKNOWN = "unknown"

_BASE_KIND = {
    "String": KIND_STRING,
    "FixedString": KIND_STRING,
    "UUID": KIND_UUID,
    "Bool": KIND_BOOL,
    "Boolean": KIND_BOOL,
    "Date": KIND_DATE,
    "Date32": KIND_DATE,
    "DateTime": KIND_DATETIME,
    "DateTime64": KIND_DATETIME,
    "Float32": KIND_FLOAT,
    "Float64": KIND_FLOAT,
    "Decimal": KIND_DECIMAL,
    "DECIMAL": KIND_DECIMAL,
    "Map": KIND_MAP,
    "Tuple": KIND_TUPLE,
}
for _bits in (8, 16, 32, 64, 128, 256):
    _BASE_KIND[f"Int{_bits}"] = KIND_INT
    _BASE_KIND[f"UInt{_bits}"] = KIND_INT

_BASE_KIND_CI = {k.lower(): v for k, v in _BASE_KIND.items()}
#: Каноническое написание базового типа: спека местами путает регистр.
_BASE_CANON = {k.lower(): k for k in _BASE_KIND}

#: Опечатки в спеке Heimdall, найденные при сборке снимка. Исправляем явным
#: списком, а не догадками: молчаливое «похожее» имя типа хуже, чем падение.
_TYPO_FIX = {"strin": "String"}

_MODIFIER = re.compile(r"\s*\((nullable|LowCardinality)\)", re.I)
_HEAD = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\((.*)\))?\s*$", re.S)


@dataclass(frozen=True)
class ChType:
    """Разобранный тип колонки."""

    raw: str
    base: str                     # String, UInt8, Array, Map, Tuple, …
    kind: str                     # одно из KIND_*
    nullable: bool = False
    low_cardinality: bool = False
    array_depth: int = 0          # 0 — скаляр, 1 — Array(...), 2 — Array(Array(...))
    args: tuple[str, ...] = field(default_factory=tuple)   # Map(String, UInt8) → ("String","UInt8")

    @property
    def is_array(self) -> bool:
        return self.array_depth > 0

    @property
    def is_scalar(self) -> bool:
        return self.array_depth == 0 and self.kind not in (KIND_MAP, KIND_TUPLE)

    @property
    def is_filterable_by_condition(self) -> bool:
        """Годится ли для узлов condition / condition_in / condition_null."""
        return self.is_scalar or self.kind == KIND_TUPLE

    @property
    def is_filterable_by_like(self) -> bool:
        """condition_like работает только по строковой скалярной колонке."""
        return self.array_depth == 0 and self.kind == KIND_STRING

    def __str__(self) -> str:  # pragma: no cover - диагностика
        return self.raw


def parse_type(raw: str) -> ChType:
    """Разобрать строку типа из спеки.

    Модификаторы ``(nullable)`` и ``(LowCardinality)`` спека пишет отдельными
    скобками после базового типа, поэтому их снимаем до разбора головы.
    """
    text = (raw or "").strip()
    nullable = False
    low_card = False

    # Снимаем модификаторы, сколько бы их ни было и в каком бы порядке.
    while True:
        m = _MODIFIER.search(text)
        if not m:
            break
        if m.group(1).lower() == "nullable":
            nullable = True
        else:
            low_card = True
        text = (text[: m.start()] + text[m.end():]).strip()

    depth = 0
    args: tuple[str, ...] = ()
    base = text
    while True:
        m = _HEAD.match(base)
        if not m:
            break
        head, inner = m.group(1), (m.group(3) or "").strip()
        if head == "Array":
            depth += 1
            if not inner:
                # Спека почти всегда пишет просто «Array» — тип элемента неизвестен.
                base = ""
                break
            base = inner
            continue
        base = head
        if inner:
            args = tuple(a.strip() for a in _split_args(inner))
        break

    # Спека местами пишет тип с опечаткой регистра («Uint8» вместо «UInt8»),
    # поэтому после точного совпадения пробуем регистронезависимое, а затем —
    # явный список известных опечаток.
    base = _TYPO_FIX.get(base.lower(), base)
    kind = _BASE_KIND.get(base)
    if kind is None:
        kind = _BASE_KIND_CI.get(base.lower(), KIND_UNKNOWN)
        if kind is not KIND_UNKNOWN:
            base = _BASE_CANON[base.lower()]
    if depth and not base:
        base = "Array"
        kind = KIND_UNKNOWN
    return ChType(raw=raw, base=base or "Unknown", kind=kind, nullable=nullable,
                  low_cardinality=low_card, array_depth=depth, args=args)


def _split_args(inner: str) -> list[str]:
    """Разбить аргументы типа по запятым верхнего уровня."""
    out, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out
