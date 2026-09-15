"""Unit tests for preflight (no LLM involved)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.benchmark.cases import BenchmarkCase
from sim.benchmark.modes import CommonConditions, build_modes
from sim.benchmark.preflight import preflight_case

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "benchmarking/cases/case-0001.json"
SCHEMA = ROOT / "benchmarking/schemas/benchmark-case-v2.schema.json"
MODEL_CATALOG = ROOT / "catalog/snapshot.json"
EMPTY_HASH = "sha256:" + "0" * 64


def skill_yaml(name: str, metric: str = "fact_count") -> str:
    return (
        f"name: {name}\n"
        f"title: {name}\n"
        "kind: recipe\nversion: 1.0.0\ndomain: org\n"
        f"description: Skill {name}\n"
        "model: {schema: dm_core, logic_model: employee_actual}\n"
        f"query: {{schema: dm_core, logic_model: employee_actual, metrics: [{metric}]}}\n"
    )


def make_catalog(path: Path, name: str = "standard", metric: str = "fact_count") -> Path:
    (path / "org").mkdir(parents=True)
    (path / "org" / f"{name}.yaml").write_text(skill_yaml(name, metric), encoding="utf-8")
    return path


def common(snapshot_id: str = "heimdall-sandbox@test") -> CommonConditions:
    return CommonConditions(
        "test-model", 0.0, "system_prompt@1", snapshot_id, True, "instant", ()
    )


def ready_case(snapshot_id: str = "heimdall-sandbox@test") -> BenchmarkCase:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update({
        "case_id": "case-ready",
        "status": "ready",
        "employee_id": "123",
        "employee_role": "manager",
        "snapshot_id": snapshot_id,
        "skill_registry_hash": EMPTY_HASH,
        "gold_answer": {
            "outcome": "answer",
            "rows": [{"grade": 10, "employee_count": 3}],
        },
    })
    return BenchmarkCase(Path("/authorial/case-ready.json"), raw)


class FakeIdentity:
    role = "manager"

    def __init__(self, _root: Path, *, hr_employee_ids: tuple[str, ...]) -> None:
        self.hr_employee_ids = hr_employee_ids

    def scope_for(self, employee_id: str) -> SimpleNamespace:
        if employee_id == "missing":
            raise KeyError(employee_id)
        return SimpleNamespace(role=self.role)


def data_snapshot(path: Path, snapshot_id: str = "heimdall-sandbox@test") -> Path:
    (path / "truth").mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8"
    )
    (path / "truth" / "people.json").write_text("{}", encoding="utf-8")
    return path


def test_preflight_skips_draft_and_generated_mock_without_llm(tmp_path: Path) -> None:
    standard = make_catalog(tmp_path / "standard")
    modes = build_modes(common(), standard, snapshots_root=tmp_path / "snapshots")
    draft_raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    draft = BenchmarkCase(EXAMPLE, draft_raw)
    draft_result = preflight_case(
        draft, modes.heimdall_skills, snapshot_root=tmp_path / "missing",
        standard_catalog_path=tmp_path / "missing", model_catalog_path=MODEL_CATALOG,
        agent_config_version="", schema_path=SCHEMA,
    )
    mock_result = preflight_case(
        ready_case(), modes.generated_skill, snapshot_root=tmp_path / "missing",
        standard_catalog_path=tmp_path / "missing", model_catalog_path=MODEL_CATALOG,
        agent_config_version="", identity_factory=FakeIdentity, schema_path=SCHEMA,
    )
    assert draft_result.status == "draft_skipped"
    assert mock_result.status == "mock_skipped"


def test_ready_preflight_validates_all_inputs_and_builds_fingerprint(tmp_path: Path) -> None:
    standard = make_catalog(tmp_path / "standard")
    modes = build_modes(common(), standard, snapshots_root=tmp_path / "snapshots")
    mode = modes.heimdall_skills
    result = preflight_case(
        ready_case(), mode, snapshot_root=data_snapshot(tmp_path / "data"),
        standard_catalog_path=mode.catalog_path, model_catalog_path=MODEL_CATALOG,
        agent_config_version="benchmark_agent@1", identity_factory=FakeIdentity,
        schema_path=SCHEMA,
    )
    assert result.status == "ready"
    assert result.fingerprint is not None
    assert result.fingerprint.data_snapshot_hash == "heimdall-sandbox@test"


def test_preflight_rejects_gold_role_and_recipe_errors(tmp_path: Path) -> None:
    standard = make_catalog(tmp_path / "standard")
    modes = build_modes(common(), standard, snapshots_root=tmp_path / "snapshots")
    mode = modes.heimdall_skills
    kwargs = {
        "snapshot_root": data_snapshot(tmp_path / "data"),
        "standard_catalog_path": mode.catalog_path,
        "model_catalog_path": MODEL_CATALOG,
        "agent_config_version": "benchmark_agent@1",
        "identity_factory": FakeIdentity,
        "schema_path": SCHEMA,
    }
    bad_gold = ready_case()
    bad_gold.raw["gold_answer"]["rows"][0]["employee_count"] = -1
    with pytest.raises(ValueError, match="gold_answer violates"):
        preflight_case(bad_gold, mode, **kwargs)
    bad_role = ready_case()
    bad_role.raw["employee_role"] = "self"
    with pytest.raises(ValueError, match="role mismatch"):
        preflight_case(bad_role, mode, **kwargs)
    missing_employee = ready_case()
    missing_employee.raw["employee_id"] = "missing"
    with pytest.raises(ValueError, match="employee does not exist"):
        preflight_case(missing_employee, mode, **kwargs)
    with pytest.raises(ValueError, match="agent_config_version must be pinned"):
        preflight_case(
            ready_case(), mode, **{**kwargs, "agent_config_version": ""}
        )

    invalid = make_catalog(tmp_path / "invalid", "bad", metric="does_not_exist")
    invalid_modes = build_modes(common(), invalid, snapshots_root=tmp_path / "invalid-snaps")
    invalid_mode = invalid_modes.heimdall_skills
    kwargs["standard_catalog_path"] = invalid_mode.catalog_path
    with pytest.raises(ValueError, match="invalid mcp_query"):
        preflight_case(ready_case(), invalid_mode, **kwargs)


def test_preflight_rejects_unavailable_snapshot_and_invalid_reference_example(
    tmp_path: Path,
) -> None:
    standard = make_catalog(tmp_path / "standard")
    modes = build_modes(common(), standard, snapshots_root=tmp_path / "snapshots")
    mode = modes.heimdall_skills
    kwargs = {
        "snapshot_root": tmp_path / "missing-data",
        "standard_catalog_path": mode.catalog_path,
        "model_catalog_path": MODEL_CATALOG,
        "agent_config_version": "benchmark_agent@1",
        "identity_factory": FakeIdentity,
        "schema_path": SCHEMA,
    }
    with pytest.raises(ValueError, match="data snapshot is unavailable"):
        preflight_case(ready_case(), mode, **kwargs)

    reference = tmp_path / "reference"
    make_catalog(reference)
    (reference / "general").mkdir()
    (reference / "general" / "bad.md").write_text(
        "---\nname: bad_reference\ntitle: Bad\nkind: reference\nversion: 1.0.0\n"
        "domain: general\ndescription: Bad JSON example\n---\n"
        "# Example\n```json\n{\"schema\": \"dm_core\", broken}\n```\n",
        encoding="utf-8",
    )
    invalid_modes = build_modes(
        common(), reference, snapshots_root=tmp_path / "reference-snaps"
    )
    kwargs.update({
        "snapshot_root": data_snapshot(tmp_path / "data"),
        "standard_catalog_path": invalid_modes.heimdall_skills.catalog_path,
    })
    with pytest.raises(ValueError, match="invalid JSON example"):
        preflight_case(
            ready_case(), invalid_modes.heimdall_skills, **kwargs
        )
