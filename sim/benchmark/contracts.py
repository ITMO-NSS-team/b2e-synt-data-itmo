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

PROMPT_RENDERER_VERSION = "benchmark-response-prompt@1"
_QUERY_PLACEHOLDER = "{{query}}"
_SCHEMA_PLACEHOLDER = "{{schema}}"


def validate_gold_contract(contract: Any) -> None:
    """Validate the public ``gold_contract`` stored in a v2 BenchmarkCase.

    Args:
        contract: Complete JSON Schema sent to the agent.

    Raises:
        ValueError: If the contract is not a valid JSON Schema requiring the
            v2 ``outcome`` and ``rows`` fields.
    """
    if not isinstance(contract, dict):
        raise ValueError("gold_contract must be an object")
    if not contract:
        raise ValueError("gold_contract must be a non-empty JSON Schema")
    try:
        Draft202012Validator.check_schema(contract)
    except Exception as exc:
        raise ValueError(f"gold_contract is invalid: {exc}") from exc
    required = set(contract.get("required", ()))
    for component in contract.get("allOf", ()):
        if isinstance(component, dict):
            required.update(component.get("required", ()))
    if not {"outcome", "rows"} <= required:
        raise ValueError("gold_contract must require outcome and rows")


def response_schema(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the complete v2 agent response schema.

    Args:
        contract: Validated public ``gold_contract``.

    Returns:
        Draft 2020-12 schema requiring ``outcome`` and ``rows``.
    """
    validate_gold_contract(contract)
    return contract


def gold_contract_hash(contract: dict[str, Any]) -> str:
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
