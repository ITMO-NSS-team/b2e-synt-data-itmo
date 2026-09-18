"""Unit tests for the public response protocol and prompt renderer."""
from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from sim.benchmark.contracts import (
    PROMPT_RENDERER_VERSION,
    render_agent_query,
    response_contract_hash,
    response_schema,
    validate_response_contract,
)
from tests.fixtures.constants import EXAMPLE


def contract() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))["response_contract"]


def test_complete_schema_accepts_result_and_explanation_without_outcome_taxonomy() -> None:
    schema = response_schema(contract())
    validator = Draft202012Validator(schema)
    validator.validate({
        "result": [{"grade": 10, "employee_count": 3}],
        "message": None,
    })
    validator.validate({
        "result": None, "message": "Не удалось получить результат",
    })
    with pytest.raises(Exception):
        validator.validate({
            "outcome": "answer", "result": None, "message": None,
        })


def test_renderer_is_deterministic_and_contains_only_public_information() -> None:
    query = "Посчитай сотрудников."
    first = render_agent_query(query, contract())
    second = render_agent_query(query, contract())
    assert first == second
    assert first.startswith(query)
    assert "grade" in first and "employee_count" in first
    assert "access_control" not in first
    assert "no_data" not in first
    assert "missing_skill" not in first
    assert "out_of_scope" not in first
    assert "expected_outcome" not in first
    assert "gold_result" not in first
    assert PROMPT_RENDERER_VERSION == "benchmark-response-prompt@1"
    assert response_contract_hash(contract()).startswith("sha256:")


def test_contract_rejects_result_schema_that_accepts_null() -> None:
    invalid = {"protocol_version": "1.0", "result_schema": {"type": ["array", "null"]}}
    with pytest.raises(ValueError, match="must not accept null"):
        validate_response_contract(invalid)
