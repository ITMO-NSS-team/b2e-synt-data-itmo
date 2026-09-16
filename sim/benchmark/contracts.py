"""Public response protocol used by every benchmark mode.

Only this module turns a business query and its public response contract into
the text sent to an agent. Gold values and evaluation rules deliberately do
not enter the renderer.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator

RESPONSE_PROTOCOL_VERSION = "1.0"
PROMPT_RENDERER_VERSION = "benchmark-response-prompt@1"


def validate_response_contract(contract: Any) -> None:
    """Validate the compact, public contract stored in a BenchmarkCase."""
    if not isinstance(contract, dict):
        raise ValueError("response_contract must be an object")
    if set(contract) != {"protocol_version", "result_schema"}:
        raise ValueError(
            "response_contract must contain only protocol_version and result_schema"
        )
    if contract["protocol_version"] != RESPONSE_PROTOCOL_VERSION:
        raise ValueError(
            f"response_contract.protocol_version must be {RESPONSE_PROTOCOL_VERSION!r}"
        )
    result_schema = contract["result_schema"]
    if not isinstance(result_schema, dict) or not result_schema:
        raise ValueError("response_contract.result_schema must be a non-empty JSON Schema")
    try:
        Draft202012Validator.check_schema(result_schema)
    except Exception as exc:
        raise ValueError(f"response_contract.result_schema is invalid: {exc}") from exc
    if Draft202012Validator(result_schema).is_valid(None):
        raise ValueError(
            "response_contract.result_schema must not accept null; null is reserved for non-answer outcomes"
        )


def response_schema(contract: dict[str, Any]) -> dict[str, Any]:
    """Expand a compact contract into the complete agent response schema."""
    validate_response_contract(contract)
    result_schema = contract["result_schema"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["result", "message"],
        "properties": {
            "result": {"anyOf": [result_schema, {"type": "null"}]},
            "message": {"type": ["string", "null"]},
        },
    }


def response_contract_hash(contract: dict[str, Any]) -> str:
    """Content hash recorded with every rendered request."""
    payload = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def render_agent_query(query: str, contract: dict[str, Any]) -> str:
    """Render the exact user message sent to the agent without gold leakage."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    schema = response_schema(contract)
    encoded = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        f"{query.strip()}\n\n"
        "Формат ответа для автоматической проверки:\n"
        "В финальном ответе обязательно приведи ровно один JSON-объект по схеме ниже. "
        "Не помещай этот объект в Markdown-блок. Если системный prompt требует "
        "дополнительный структурированный хвост, выполни и это требование отдельно.\n"
        "Если задача выполнена, запиши структурированный результат в result. "
        "Если результат получить нельзя, верни result: null и кратко объясни "
        "причину в message. При успешном ответе message может быть null.\n"
        "Ответ должен соответствовать JSON Schema:\n"
        f"{encoded}"
    )
