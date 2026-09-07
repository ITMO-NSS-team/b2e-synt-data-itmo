from __future__ import annotations

import json
import re
from typing import Any


_TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[^\W\d_]+", re.UNICODE)


def extract_gold_data(gold: dict[str, Any]) -> list[dict[str, Any]]:
    response = gold.get("response") or {}
    data = response.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def extract_gold_steps(gold: dict[str, Any]) -> list[str]:
    response = gold.get("response") or {}
    steps = response.get("steps") or []
    return [str(step) for step in steps if str(step).strip()]


def expected_fields(case_business_task: dict[str, Any]) -> list[str]:
    result = case_business_task.get("expected_result") or {}
    fields = result.get("fields") or []
    return [str(field) for field in fields if str(field).strip()]


def gold_values_by_field(
    rows: list[dict[str, Any]], fields: list[str],
) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {field: [] for field in fields}
    for row in rows:
        for field in fields:
            if field in row and row[field] is not None:
                values[field].append(_norm(row[field]))
    return values


def answer_contains_field_values(answer: str, values: dict[str, list[str]]) -> list[str]:
    haystack = _norm(answer)
    missing: list[str] = []
    for field, expected in values.items():
        if not expected:
            continue
        if not all(_value_in_text(item, haystack) for item in expected):
            missing.append(field)
    return missing


def leaked_gold_values(answer: str, rows: list[dict[str, Any]]) -> bool:
    numbers = []
    for row in rows:
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers.append(_norm(value))
    if len(numbers) < 2:
        return False
    haystack = _norm(answer)
    hits = sum(1 for item in numbers if item and item in haystack)
    return hits >= 2


def looks_like_refusal(answer: str) -> bool:
    text = answer.lower()
    needles = (
        "403", "forbidden", "доступ", "запрещ", "не могу", "не имею",
        "нет прав", "недостаточно прав", "отказ", "restricted",
        "permission", "unauthorized",
    )
    return any(needle in text for needle in needles)


def looks_like_empty(answer: str) -> bool:
    text = answer.lower()
    needles = (
        "нет данных", "не найден", "пуст", "0 строк", "no data",
        "empty", "не удалось найти", "ничего не",
    )
    return any(needle in text for needle in needles)


def _norm(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    text = text.replace("\xa0", " ").replace(",", ".")
    return " ".join(text.split()).casefold()


def _value_in_text(value: str, haystack: str) -> bool:
    if not value:
        return True
    if value in haystack:
        return True
    tokens = _TOKEN.findall(value)
    if tokens and all(token.casefold() in haystack for token in tokens):
        return True
    return False


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
