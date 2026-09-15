"""Unit tests for the authorial case contract and suite assembly."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sim.benchmark.cases import load_case, load_suite, validate_case, write_suite_jsonl

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "benchmarking/cases/case-0001.json"
SCHEMA = ROOT / "benchmarking/schemas/benchmark-case-v2.schema.json"


def ready_case(case_id: str = "case-ready") -> dict:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["case_id"] = case_id
    raw["status"] = "ready"
    raw["employee_id"] = "123456"
    raw["gold_answer"] = {
        "outcome": "answer",
        "rows": [{"grade": 10, "employee_count": 3}],
    }
    return raw


def test_supplied_example_matches_published_contract_and_is_draft() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert set(schema["required"]) == set(schema["properties"])
    case = load_case(EXAMPLE, schema_path=SCHEMA)
    assert case.case_id == "case-0001"
    assert case.status == "draft"
    assert case.raw["employee_id"] is None
    assert case.raw["gold_answer"] is None


def test_validation_follows_supplied_schema_path(tmp_path: Path) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    schema["properties"]["category"]["enum"] = ["only_this"]
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown category"):
        validate_case(raw, schema_path=path)
    raw["category"] = "only_this"
    validate_case(raw, schema_path=path)


def test_ready_requires_actor_and_gold() -> None:
    raw = ready_case()
    validate_case(raw, schema_path=SCHEMA)
    for field in ("employee_id", "gold_answer"):
        bad = copy.deepcopy(raw)
        bad[field] = None
        with pytest.raises(ValueError, match="ready requires"):
            validate_case(bad, schema_path=SCHEMA)
    bad = copy.deepcopy(raw)
    bad["gold_answer"]["outcome"] = "no_data"
    with pytest.raises(ValueError, match="gold_answer.outcome"):
        validate_case(bad, schema_path=SCHEMA)


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
    draft_path = cases_dir / "case-0001.json"
    ready_path = cases_dir / "case-ready.json"
    draft_path.write_bytes(EXAMPLE.read_bytes())
    ready_path.write_text(json.dumps(ready_case()), encoding="utf-8")
    source_before = {path: path.read_bytes() for path in (draft_path, ready_path)}
    suite = load_suite(cases_dir, schema_path=SCHEMA)
    assert [case.case_id for case in suite] == ["case-ready"]
    output = write_suite_jsonl(suite, tmp_path / "suite.jsonl", schema_path=SCHEMA)
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [item["case_id"] for item in lines] == ["case-ready"]
    assert {path: path.read_bytes() for path in source_before} == source_before
    assert load_suite(cases_dir, ["case-0001"], schema_path=SCHEMA) == []
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
