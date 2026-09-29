from __future__ import annotations

import json
from typing import Any


def retrieved_skills(trace: dict[str, Any] | None) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            attributes = value.get("attributes") or {}
            arguments = (
                _decode(_attr(attributes, "input.value"))
                or _decode(_attr(attributes, "tool.parameters"))
                or _decode(_attr(attributes, "tool_parameters"))
            )
            tool_name = str(
                value.get("name")
                or _attr(attributes, "tool.name")
                or ""
            )
            if tool_name.endswith("get_skill"):
                if isinstance(arguments, dict) and arguments.get("name"):
                    found.append(str(arguments["name"]))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(trace)
    return sorted(set(found))


def root_span_id(trace: dict[str, Any] | None) -> str | None:
    if not trace:
        return None
    spans = trace.get("spans") or trace.get("tree") or []
    if isinstance(spans, list) and spans:
        first = spans[0]
        if isinstance(first, dict):
            return (str(first.get("span_id") or "")
                    or str((first.get("context") or {}).get("span_id") or "")
                    or None)
    return None


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _attr(attributes: dict[str, Any], name: str) -> Any:
    if name in attributes:
        return attributes[name]
    current: Any = attributes
    for part in name.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
