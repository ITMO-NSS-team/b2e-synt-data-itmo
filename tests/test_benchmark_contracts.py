"""Unit tests for the public response protocol and prompt renderer."""
from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from sim.benchmark.contracts import (
    PROMPT_RENDERER_VERSION, gold_contract_hash,
    render_agent_query,
    response_schema,
    validate_gold_contract,
)
from sim.benchmark.path_lib import RESPONSE_PROMPT_PATH
from tests.fixtures.constants import EXAMPLE


def contract() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))["gold_contract"]


def test_complete_schema_accepts_v2_outcome_and_rows() -> None:
    schema = response_schema(contract())
    validator = Draft202012Validator(schema)
    validator.validate({
        "outcome": "answer",
        "rows": [{"grade": 10, "employee_count": 3}],
    })
    with pytest.raises(Exception):
        validator.validate({
            "result": [], "message": None,
        })


def test_renderer_is_deterministic_and_contains_only_public_information() -> None:
    query = "Посчитай сотрудников."
    first = render_agent_query(query, contract())
    second = render_agent_query(query, contract())
    assert first == second
    assert first.startswith(query)
    assert "grade" in first and "employee_count" in first
    assert '"outcome"' in first and '"rows"' in first
    assert "expected_outcome" not in first
    assert '"employee_count": 3' not in first
    assert PROMPT_RENDERER_VERSION == "benchmark-response-prompt@1"
    assert "{{query}}" in RESPONSE_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{schema}}" in RESPONSE_PROMPT_PATH.read_text(encoding="utf-8")
    assert gold_contract_hash(contract()).startswith("sha256:")


def test_contract_requires_outcome_and_rows() -> None:
    invalid = {"type": "object", "required": ["rows"], "properties": {"rows": {}}}
    with pytest.raises(ValueError, match="require outcome and rows"):
        validate_gold_contract(invalid)
