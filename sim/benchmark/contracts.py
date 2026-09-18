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

from .path_lib import RESPONSE_PROMPT_PATH

RESPONSE_PROTOCOL_VERSION = "1.0"
PROMPT_RENDERER_VERSION = "benchmark-response-prompt@1"
_QUERY_PLACEHOLDER = "{{query}}"
_SCHEMA_PLACEHOLDER = "{{schema}}"


def validate_response_contract(contract: Any) -> None:
    """Validate the compact public contract stored in a BenchmarkCase.

    Args:
        contract: Object with ``protocol_version`` and ``result_schema``.

    Raises:
        ValueError: If the contract shape or JSON Schema is invalid, or if
            ``result_schema`` accepts ``null`` (reserved for non-answer outcomes).
    """
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
    """Expand a compact contract into the complete agent response schema.

    Args:
        contract: Validated public response contract.

    Returns:
        Draft 2020-12 schema requiring ``result`` and ``message``.
    """
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
    """Content hash recorded with every rendered request.

    Args:
        contract: Public response contract as stored on the case.

    Returns:
        Canonical ``sha256:<hex>`` digest of the sorted JSON payload.
    """
    payload = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def render_agent_query(query: str, contract: dict[str, Any]) -> str:
    """Render the exact user message sent to the agent without gold leakage.

    Args:
        query: Original business question from the case.
        contract: Public response contract; evaluation gold is not used.

    Returns:
        User text with the JSON Schema the agent must satisfy.

    Raises:
        ValueError: If ``query`` is empty or the contract is invalid.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    schema = response_schema(contract)
    encoded = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        _response_prompt_template()
        .replace(_QUERY_PLACEHOLDER, query.strip())
        .replace(_SCHEMA_PLACEHOLDER, encoded)
    )


def _response_prompt_template() -> str:
    text = RESPONSE_PROMPT_PATH.read_text(encoding="utf-8")
    if _QUERY_PLACEHOLDER not in text or _SCHEMA_PLACEHOLDER not in text:
        raise ValueError(
            f"{RESPONSE_PROMPT_PATH} must contain {_QUERY_PLACEHOLDER} and "
            f"{_SCHEMA_PLACEHOLDER}"
        )
    return text.rstrip("\n")
