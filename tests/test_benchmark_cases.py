"""Unit tests for the authorial case contract and suite assembly."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sim.benchmark.cases import load_case, load_suite, validate_case, write_suite_jsonl
from tests.fixtures.constants import EXAMPLE, SCHEMA


def ready_case(case_id: str = "case-9999") -> dict:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["case_id"] = case_id
    raw["status"] = "verified"
    raw["employee_id"] = 123456
    return raw


def test_supplied_example_matches_published_contract_and_is_ready() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert set(schema["required"]) == set(schema["properties"])
    case = load_case(EXAMPLE, schema_path=SCHEMA)
    assert case.case_id == "case-0000"
    assert case.status == "verified"
    assert case.raw["employee_id"] == 123456
    assert case.raw["gold_answer"]["outcome"] == "answer"

    incomplete = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    incomplete["status"] = "draft"
    incomplete["gold_contract"] = {}
    incomplete["gold_answer"] = None
    incomplete["employee_id"] = None
    validate_case(incomplete, schema_path=SCHEMA)


def test_validation_follows_supplied_schema_path(tmp_path: Path) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    schema["properties"]["category"]["enum"] = ["only_this"]
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["status"] = "draft"
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown category"):
        validate_case(raw, schema_path=path)
    raw["category"] = "only_this"
    validate_case(raw, schema_path=path)


def test_v2_loader_resolves_local_contract_and_comparison_template(tmp_path: Path) -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    contract = raw["gold_contract"]
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "gold-contract.json").write_text(
        json.dumps(contract), encoding="utf-8",
    )
    raw["gold_contract"] = {
        "$ref": "templates/gold-contract.json",
        "properties": {
            "rows": contract["properties"]["rows"],
        },
    }
    raw["gold_comparison"] = {
        "template": "simple_comparison",
        "row_key": ["grade"],
    }
    path = tmp_path / "case-0000.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    case = load_case(path, schema_path=SCHEMA)

    assert "$ref" not in case.raw["gold_contract"]
    assert case.raw["gold_contract"]["allOf"]
    assert case.raw["gold_comparison"] == {
        "ordered": False,
        "row_key": ["grade"],
        "allow_extra_rows": False,
        "numeric_absolute_tolerance": 0,
    }


def test_ready_requires_actor_and_gold() -> None:
    raw = ready_case()
    validate_case(raw, schema_path=SCHEMA)
    for field in ("employee_id", "gold_answer"):
        bad = copy.deepcopy(raw)
        bad[field] = None
        with pytest.raises(ValueError, match="verified requires"):
            validate_case(bad, schema_path=SCHEMA)
    bad = copy.deepcopy(raw)
    bad["gold_answer"]["outcome"] = "no_data"
    bad["gold_answer"]["rows"] = []
    with pytest.raises(ValueError, match="incompatible with category"):
        validate_case(bad, schema_path=SCHEMA)


def test_missing_skill_can_expect_base_tool_answer_or_honest_refusal() -> None:
    fallback = ready_case()
    fallback["category"] = "missing_skill"
    validate_case(fallback, schema_path=SCHEMA)
    refusal = copy.deepcopy(fallback)
    refusal["gold_answer"]["outcome"] = "missing_skill"
    refusal["gold_answer"]["rows"] = []
    refusal["gold_comparison"]["row_key"] = []
    validate_case(refusal, schema_path=SCHEMA)


def test_rejects_invalid_authorial_fields() -> None:
    baseline = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    bad = copy.deepcopy(baseline)
    bad["expected_skills"] = "headcount_by_dimension"
    with pytest.raises(ValueError, match="expected_skills"):
        validate_case(bad, schema_path=SCHEMA)
    bad = copy.deepcopy(baseline)
    bad["employee_idd"] = bad.pop("employee_id")
    with pytest.raises(ValueError, match="case fields"):
        validate_case(bad, schema_path=SCHEMA)
    bad = copy.deepcopy(baseline)
    bad["status"] = []
    with pytest.raises(ValueError, match="status must be"):
        validate_case(bad, schema_path=SCHEMA)


def test_suite_skips_draft_and_preserves_source_json(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    draft_path = cases_dir / "case-fixture.json"
    ready_path = cases_dir / "case-9999.json"
    draft = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    draft["status"] = "draft"
    draft["employee_id"] = None
    draft["gold_answer"] = None
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    ready_path.write_text(json.dumps(ready_case()), encoding="utf-8")
    source_before = {path: path.read_bytes() for path in (draft_path, ready_path)}
    suite = load_suite(cases_dir, schema_path=SCHEMA)
    assert [case.case_id for case in suite] == ["case-9999"]
    output = write_suite_jsonl(suite, tmp_path / "suite.jsonl", schema_path=SCHEMA)
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [item["case_id"] for item in lines] == ["case-9999"]
    assert {path: path.read_bytes() for path in source_before} == source_before
    assert load_suite(cases_dir, ["case-0000"], schema_path=SCHEMA) == []
    with pytest.raises(ValueError, match="draft case"):
        load_suite(cases_dir, on_draft="error", schema_path=SCHEMA)
    with pytest.raises(ValueError, match="unknown case_ids"):
        load_suite(cases_dir, ["missing"], schema_path=SCHEMA)


def test_suite_refuses_duplicate_ids_and_source_overwrite(tmp_path: Path) -> None:
    raw = ready_case()
    for name in ("first.json", "second.json"):
        (tmp_path / name).write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_suite(tmp_path, schema_path=SCHEMA)
    (tmp_path / "second.json").unlink()
    suite = load_suite(tmp_path, schema_path=SCHEMA)
    with pytest.raises(ValueError, match="cannot overwrite"):
        write_suite_jsonl(suite, tmp_path / "first.json", schema_path=SCHEMA)
    with pytest.raises(ValueError, match=".jsonl extension"):
        write_suite_jsonl(suite, tmp_path / "other.json", schema_path=SCHEMA)
    suite[0].raw["query"] = None
    with pytest.raises(ValueError, match="query must be"):
        write_suite_jsonl(suite, tmp_path / "mutated.jsonl", schema_path=SCHEMA)
