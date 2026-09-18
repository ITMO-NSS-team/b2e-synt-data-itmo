"""Input and live-condition handling for the benchmark command."""
from __future__ import annotations

import json

import pytest

from sim.benchmark.cli import common_conditions, load_cases_path, select_modes
from sim.benchmark.modes import CommonConditions, build_modes
from tests.fixtures.constants import EXAMPLE


def live_payload() -> dict:
    config = {
        "model_id": "light-model", "temperature": 0.0,
        "code_execution": "forbidden", "conversation_mode": "stateless",
        "tool_subset": [
            "list_models", "describe_model", "get_docs", "mcp_query",
            "find_skills", "get_skill",
        ],
    }
    off = dict(config)
    off["tool_subset"] = [
        name for name in config["tool_subset"]
        if name not in {"find_skills", "get_skill"}
    ]
    return {
        "refs": {
            "skills_disabled": "benchmark_off@1",
            "heimdall_skills": "benchmark_on@1",
        },
        "configs": {
            "benchmark_off@1": off,
            "benchmark_on@1": config,
        },
        "prompt_versions": {
            "benchmark_off@1": "system_prompt@2",
            "benchmark_on@1": "system_prompt@2",
        },
        "emulator": {
            "data_snapshot_hash": "snapshot@test",
            "traps_enabled": True,
            "latency_profile": "instant",
            "hr_employee_ids": [],
        },
    }


def test_cases_path_accepts_directory_json_and_jsonl_and_skips_draft(tmp_path) -> None:
    ready = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    draft = dict(ready)
    draft.update({
        "case_id": "case-draft", "status": "draft", "employee_id": None,
        "evaluation_contract": None,
    })
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "ready.json").write_text(json.dumps(ready), encoding="utf-8")
    (cases / "draft.json").write_text(json.dumps(draft), encoding="utf-8")
    assert [item.case_id for item in load_cases_path(cases)] == ["case-fixture"]
    assert [item.case_id for item in load_cases_path(cases / "ready.json")] == ["case-fixture"]
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(draft) + "\n" + json.dumps(ready) + "\n",
        encoding="utf-8",
    )
    assert [item.case_id for item in load_cases_path(suite)] == ["case-fixture"]


def test_case_limit_selects_first_ready_cases_after_stable_sort(tmp_path) -> None:
    template = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    cases = tmp_path / "cases"
    cases.mkdir()
    for case_id in ("case-003", "case-001", "case-002"):
        raw = dict(template)
        raw["case_id"] = case_id
        (cases / f"{case_id}.json").write_text(
            json.dumps(raw), encoding="utf-8",
        )

    selected = load_cases_path(cases, limit=2)

    assert [item.case_id for item in selected] == ["case-001", "case-002"]


def test_case_limit_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="limit must be >= 1"):
        load_cases_path(EXAMPLE, limit=0)


def test_live_manifest_builds_common_conditions_and_rejects_drift(tmp_path) -> None:
    payload = live_payload()
    common = common_conditions(payload)
    assert common.model_id == "light-model"
    assert common.prompt_registry_version == "system_prompt@2"
    assert common.snapshot_id == "snapshot@test"

    payload["configs"]["benchmark_off@1"]["temperature"] = 0.7
    with pytest.raises(ValueError, match="differ by temperature"):
        common_conditions(payload)


def test_generated_mock_cannot_be_selected(tmp_path) -> None:
    base = tmp_path / "skills"
    (base / "org").mkdir(parents=True)
    (base / "org" / "one.yaml").write_text(
        "name: one\ntitle: One\nkind: recipe\ndomain: org\n"
        "description: One\nmodel: {schema: dm_core, logic_model: employee_actual}\n"
        "query: {schema: dm_core, logic_model: employee_actual, metrics: [fact_count]}\n",
        encoding="utf-8",
    )
    common = CommonConditions(
        "light-model", 0.0, "system_prompt@2", "snapshot@test", True,
        "instant", (),
    )
    modes = build_modes(common, base, snapshots_root=tmp_path / "snapshots")
    selected = select_modes(modes, "skills_disabled,heimdall_skills")
    assert set(selected) == {"skills_disabled", "heimdall_skills"}
    with pytest.raises(ValueError, match="still a mock"):
        select_modes(modes, "generated_skill")
