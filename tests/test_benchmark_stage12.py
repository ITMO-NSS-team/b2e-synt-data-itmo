"""Unit tests for the case contract, suite assembly and three mode configs."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sim.benchmark.cases import load_case, load_suite, validate_case, write_suite_jsonl
from sim.benchmark.modes import (
    DATA_TOOLS, SKILL_TOOLS, BenchmarkMode, CommonConditions, build_modes,
    catalog_hash, write_mode_config,
)

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


def skill_yaml(name: str) -> str:
    return (
        f"name: {name}\n"
        f"title: {name}\n"
        "kind: recipe\n"
        "domain: org\n"
        f"description: Skill {name}\n"
        "model: {schema: dm_core, logic_model: employee_actual}\n"
        "query: {schema: dm_core, logic_model: employee_actual, metrics: [fact_count]}\n"
    )


@pytest.fixture()
def catalogs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "heimdall-skills"
    generated = tmp_path / "generated"
    (base / "org").mkdir(parents=True)
    (generated / "org").mkdir(parents=True)
    (base / "org" / "standard.yaml").write_text(skill_yaml("standard"), encoding="utf-8")
    (generated / "org" / "generated.yaml").write_text(skill_yaml("generated"), encoding="utf-8")
    return base, generated


@pytest.fixture()
def common() -> CommonConditions:
    return CommonConditions(
        model_id="test-model",
        temperature=0.0,
        prompt_registry_version="system_prompt@1",
        snapshot_id="heimdall-sandbox@test",
        traps_enabled=True,
        latency_profile="instant",
        hr_employee_ids=("999999",),
    )


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


def test_three_modes_pin_same_common_conditions(
    catalogs: tuple[Path, Path], common: CommonConditions,
) -> None:
    base, generated = catalogs
    modes = build_modes(common, base, generated)
    assert set(modes) == {mode.value for mode in BenchmarkMode}
    assert all(mode.common is common for mode in modes.values())
    assert modes[BenchmarkMode.SKILLS_DISABLED].tool_subset == DATA_TOOLS
    assert modes[BenchmarkMode.SKILLS_DISABLED].catalog_path is None
    assert modes[BenchmarkMode.HEIMDALL_SKILLS].tool_subset == SKILL_TOOLS
    assert modes[BenchmarkMode.GENERATED_SKILL].tool_subset == SKILL_TOOLS
    assert modes[BenchmarkMode.HEIMDALL_SKILLS].catalog_hash == catalog_hash(base)
    assert modes[BenchmarkMode.GENERATED_SKILL].catalog_hash != (
        modes[BenchmarkMode.HEIMDALL_SKILLS].catalog_hash
    )
    assert modes[BenchmarkMode.GENERATED_SKILL].generated_skill_names == ("generated",)


def test_config_file_explicitly_contains_all_required_variables(
    tmp_path: Path, catalogs: tuple[Path, Path], common: CommonConditions,
) -> None:
    modes = build_modes(common, *catalogs)
    path = write_mode_config(modes, tmp_path / "modes.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == "1.0"
    for mode in saved["modes"].values():
        assert set(mode["common"]) == {
            "model_id", "temperature", "prompt_registry_version", "snapshot_id",
            "traps_enabled", "latency_profile", "hr_employee_ids", "code_execution",
        }
        assert mode["common"]["code_execution"] == "forbidden"
        assert "tool_subset" in mode
        assert "catalog_path" in mode
        assert "catalog_hash" in mode
    assert saved["modes"][BenchmarkMode.GENERATED_SKILL]["generated_skill_names"] == ["generated"]


def test_hash_changes_only_when_skill_files_change(catalogs: tuple[Path, Path]) -> None:
    base, _generated = catalogs
    original = catalog_hash(base)
    (base / "README.md").write_text("irrelevant docs", encoding="utf-8")
    assert catalog_hash(base) == original
    skill = base / "org" / "standard.yaml"
    skill.write_text(skill.read_text(encoding="utf-8") + "notes: [changed]\n", encoding="utf-8")
    assert catalog_hash(base) != original


def test_collision_and_invalid_catalog_are_refused(
    catalogs: tuple[Path, Path], common: CommonConditions,
) -> None:
    base, generated = catalogs
    skill = generated / "org" / "generated.yaml"
    skill.write_text(skill_yaml("standard"), encoding="utf-8")
    with pytest.raises(ValueError, match="collide"):
        build_modes(common, base, generated)
    skill.write_text("name: bad\nkind: recipe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid skill catalog"):
        build_modes(common, base, generated)


def test_unpinned_prompt_and_code_execution_are_refused() -> None:
    with pytest.raises(ValueError, match="pinned"):
        CommonConditions("model", 0.0, "system_prompt", "snapshot", True, "instant", ())
    with pytest.raises(ValueError, match="forbidden"):
        CommonConditions("model", 0.0, "system_prompt@1", "snapshot", True, "instant", (),
                         code_execution="allowed")
    with pytest.raises(ValueError, match="finite"):
        CommonConditions("model", float("nan"), "system_prompt@1", "snapshot", True,
                         "instant", ())


def test_writer_rejects_mismatched_common_conditions(
    tmp_path: Path, catalogs: tuple[Path, Path], common: CommonConditions,
) -> None:
    modes = build_modes(common, *catalogs)
    other = CommonConditions("other-model", 0.0, "system_prompt@1", "heimdall-sandbox@test",
                             True, "instant", ("999999",))
    modes[BenchmarkMode.GENERATED_SKILL] = replace(
        modes[BenchmarkMode.GENERATED_SKILL], common=other)
    with pytest.raises(ValueError, match="identical common conditions"):
        write_mode_config(modes, tmp_path / "invalid.json")
