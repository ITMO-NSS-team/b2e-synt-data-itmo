"""Unit tests for the three mode configs and catalog hashes."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sim.benchmark.modes import (
    DATA_TOOLS, SKILL_TOOLS, BenchmarkMode, CommonConditions, build_modes,
    catalog_hash, write_mode_config,
)


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


def test_three_modes_pin_same_common_conditions(
    catalogs: tuple[Path, Path], common: CommonConditions,
) -> None:
    base, generated = catalogs
    modes = build_modes(common, base, generated, snapshots_root=base.parent / "snapshots")
    assert {mode.name for mode in modes} == {item.value for item in BenchmarkMode}
    assert all(mode.common is common for mode in modes)
    assert modes.skills_disabled.tool_subset == DATA_TOOLS
    assert modes.skills_disabled.catalog_path is None
    assert modes.heimdall_skills.tool_subset == SKILL_TOOLS
    assert modes.generated_skill.tool_subset == SKILL_TOOLS
    assert modes.heimdall_skills.catalog_hash == catalog_hash(base)
    assert modes.generated_skill.catalog_hash != modes.heimdall_skills.catalog_hash
    assert modes.generated_skill.generated_skill_names == ("generated",)


def test_config_file_explicitly_contains_all_required_variables(
    tmp_path: Path, catalogs: tuple[Path, Path], common: CommonConditions,
) -> None:
    modes = build_modes(common, *catalogs, snapshots_root=tmp_path / "snapshots")
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
        build_modes(common, base, generated, snapshots_root=base.parent / "snapshots")
    skill.write_text("name: bad\nkind: recipe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid skill catalog"):
        build_modes(common, base, generated, snapshots_root=base.parent / "snapshots")


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
    modes = build_modes(common, *catalogs, snapshots_root=tmp_path / "snapshots")
    other = CommonConditions("other-model", 0.0, "system_prompt@1", "heimdall-sandbox@test",
                             True, "instant", ("999999",))
    modes = replace(modes, generated_skill=replace(modes.generated_skill, common=other))
    with pytest.raises(ValueError, match="identical common conditions"):
        write_mode_config(modes, tmp_path / "invalid.json")


def test_disabled_and_mock_generated_modes_are_explicit(
    tmp_path: Path, common: CommonConditions,
) -> None:
    base = tmp_path / "standard"
    (base / "org").mkdir(parents=True)
    (base / "org" / "standard.yaml").write_text(skill_yaml("standard"), encoding="utf-8")
    modes = build_modes(common, base, snapshots_root=tmp_path / "snapshots")
    assert modes.skills_disabled.skills_enabled is False
    assert "find_skills" not in modes.skills_disabled.tool_subset
    assert "get_skill" not in modes.skills_disabled.tool_subset
    assert modes.generated_skill.is_mock is True
    assert modes.generated_skill.generated_skill_names == ()


def test_generated_mode_builds_combined_snapshot_and_rejects_duplicates(
    tmp_path: Path, catalogs: tuple[Path, Path], common: CommonConditions,
) -> None:
    base, generated = catalogs
    modes = build_modes(common, base, generated, snapshots_root=tmp_path / "snapshots")
    mode = modes.generated_skill
    assert mode.is_mock is False
    assert {path.stem for path in Path(mode.catalog_path).rglob("*.yaml")} == {
        "standard", "generated",
    }
    duplicate = tmp_path / "duplicate"
    (duplicate / "org").mkdir(parents=True)
    (duplicate / "org" / "standard.yaml").write_text(skill_yaml("standard"), encoding="utf-8")
    with pytest.raises(ValueError, match="collide"):
        build_modes(common, base, duplicate, snapshots_root=tmp_path / "other")
