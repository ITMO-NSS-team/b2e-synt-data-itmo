from __future__ import annotations

import json
from pathlib import Path

import pytest

from heimdall.skills.registry import Registry

from sim.benchmark.generated_plan import _expected_skill, build_generated_plan
from tests.fixtures.constants import EXAMPLE, SCHEMA


def _skill(root: Path, name: str, *, variant_id: str | None = None, version: str = "0.1.0") -> None:
    path = root / (variant_id or name) / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"title: {name}\n"
        "kind: reference\n"
        "domain: test\n"
        f"description: Generated {name}\n"
        f"version: {version}\n"
        "status: active\n"
        "---\n\nUse the available data tools.\n",
        encoding="utf-8",
    )
    (path.parent / "resource.txt").write_text("supporting context", encoding="utf-8")


def _case(
    root: Path,
    case_id: str,
    expected: list[str],
) -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["case_id"] = case_id
    raw["expected_skills"] = expected
    (root / f"{case_id}.json").write_text(json.dumps(raw), encoding="utf-8")


def test_plan_selects_covered_cases_once_and_expands_generated_variants(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha"])
    _case(cases, "case-0101-01", ["alpha"])
    _case(cases, "case-0102-00", ["missing"])
    _case(cases, "case-0103-00", [])
    _case(cases, "case-0104-00", ["alpha", "beta"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha", variant_id="alpha_1")
    _skill(generated, "alpha", variant_id="alpha_2")
    _skill(generated, "unused")

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA,
    )

    assert plan["selected_case_ids"] == [
        "case-0101-00", "case-0101-01", "case-0104-00",
    ]
    covered = [json.loads(line) for line in Path(plan["covered_cases_path"]).read_text().splitlines()]
    assert [row["case_id"] for row in covered] == [
        "case-0101-00", "case-0101-01", "case-0104-00",
    ]
    assert len(plan["groups"]) == 2
    assert [group["variant_id"] for group in plan["groups"]] == ["alpha_1", "alpha_2"]
    for group in plan["groups"]:
        assert group["skill_names"] == ["alpha"]
        assert group["expected_skills"] == ["alpha"]
        assert group["replaces_existing"] is False
        assert group["case_ids"] == [
            "case-0101-00", "case-0101-01", "case-0104-00",
        ]
        isolated = Registry.load(group["catalog_path"])
        assert isolated.active() == ["alpha"]
        assert (Path(group["catalog_path"]) / "alpha" / "resource.txt").read_text() == (
            "supporting context"
        )
    assert [row["status"] for row in plan["excluded"]] == [
        "generated_skill_unavailable_excluded",
        "generated_no_expected_skill_excluded",
    ]
    assert plan["partially_covered"] == [{
        "case_id": "case-0104-00",
        "available_skills": ["alpha"],
        "unavailable_skills": ["beta"],
    }]


def test_named_catalogs_match_metadata_and_keep_every_variant(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha-v1-0-1"])
    _case(cases, "case-0102-00", ["missing-v1-0-0"])
    baseline = tmp_path / "baseline"
    _skill(baseline, "alpha-compact-aaaaaaaaaa-bbbbbbbbbb",
           variant_id="compact/medium/first", version="1.0.1")
    sf = tmp_path / "skillfactory"
    sf.mkdir()
    (sf / "second.md").write_text(
        "---\nname: alpha\ntitle: Alpha\nkind: reference\ndomain: test\n"
        "description: Alpha\nversion: 1.0.1\nstatus: active\n---\n\nUse data tools.\n",
        encoding="utf-8",
    )
    _skill(sf, "unmatched", variant_id="unmatched", version="1.0.0")

    plan = build_generated_plan(
        cases, None, tmp_path / "work", schema_path=SCHEMA,
        generated_sources={"baseline": baseline, "skillfactory": sf},
    )

    assert plan["selected_case_ids"] == ["case-0101-00"]
    assert [(group["source"], group["version"], group["case_ids"])
            for group in plan["groups"]] == [
        ("baseline", "1.0.1", ["case-0101-00"]),
        ("skillfactory", "1.0.1", ["case-0101-00"]),
    ]
    assert len({group["variant_id"] for group in plan["groups"]}) == 2
    assert plan["groups"][0]["path_labels"] == ["compact", "medium"]
    assert plan["groups"][1]["path_labels"] == []
    assert [row["skill_name"] for row in plan["unmatched_artifacts"]] == ["unmatched"]
    assert plan["excluded"][0]["case_id"] == "case-0102-00"
    assert not (tmp_path / "work/catalogs/skillfactory/generated-only/skillfactory--unmatched").exists()


def test_plan_records_zero_coverage_without_copying_skills(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["missing"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha")

    plan = build_generated_plan(cases, generated, tmp_path / "work", schema_path=SCHEMA)

    assert plan["selected_case_ids"] == []
    assert plan["groups"] == []
    assert plan["unmatched_artifacts"][0]["skill_name"] == "alpha"
    assert Path(plan["covered_cases_path"]).read_text(encoding="utf-8") == ""
    assert not list((tmp_path / "work/catalogs").rglob("SKILL.md"))


def test_plan_resume_checks_catalog_content(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha")
    work = tmp_path / "work"
    first = build_generated_plan(cases, generated, work, schema_path=SCHEMA)
    assert build_generated_plan(cases, generated, work, schema_path=SCHEMA, resume=True) == first
    (generated / "alpha" / "resource.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="resume inputs differ"):
        build_generated_plan(cases, generated, work, schema_path=SCHEMA, resume=True)


def test_limit_is_applied_after_generated_coverage_filter(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["missing"])
    _case(cases, "case-0102-00", ["alpha"])
    _case(cases, "case-0103-00", ["beta"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha")
    _skill(generated, "beta")

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA, limit=1,
    )

    assert plan["eligible_case_ids"] == ["case-0102-00", "case-0103-00"]
    assert plan["selected_case_ids"] == ["case-0102-00"]
    assert [row["skill_name"] for row in plan["unmatched_artifacts"]] == ["beta"]
    assert not (tmp_path / "work/catalogs/generated/generated-only/beta").exists()


def test_plan_matches_hashed_skill_by_its_logical_name(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha"])

    generated_name = "alpha-compact-a94e814444-b4d7da560f"
    generated = tmp_path / "generated"
    _skill(generated, generated_name)

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA,
    )

    assert plan["selected_case_ids"] == ["case-0101-00"]
    assert len(plan["groups"]) == 1
    assert plan["groups"][0]["expected_skills"] == ["alpha"]
    assert plan["groups"][0]["skill_names"] == [generated_name]
    generated_case = json.loads(
        Path(plan["groups"][0]["cases_path"]).read_text(encoding="utf-8")
    )
    assert generated_case["expected_skills"] == [generated_name]


def test_generated_name_parser_keeps_legacy_compatibility() -> None:
    assert _expected_skill("alpha-full-a94e814444-b4d7da560f") == "alpha"
    assert _expected_skill("alpha-full-a94e814444") == "alpha"
    assert _expected_skill("alpha-full-not-a-hash") == "alpha-full-not-a-hash"


def test_multiple_expected_skills_run_individually_by_default(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha", "beta"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha")
    _skill(generated, "beta")

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA,
    )

    assert [group["skill_names"] for group in plan["groups"]] == [
        ["alpha"], ["beta"],
    ]


def test_plan_builds_existing_plus_one_generated_catalog(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha", variant_id="alpha_1")
    _skill(generated, "unused")
    base = tmp_path / "base"
    _skill(base, "standard")

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA,
        base_catalog_path=base,
        generated_modes=("generated_skills", "existing_plus_generated"),
    )

    assert plan["selected_case_ids"] == ["case-0101-00"]
    assert plan["generated_modes"] == ["generated_skills", "existing_plus_generated"]
    assert len(plan["groups"]) == 1
    group = plan["groups"][0]
    assert group["replaces_existing"] is False
    assert Registry.load(group["catalog_path"]).active() == ["alpha"]
    assert Registry.load(group["combined_catalog_path"]).active() == ["alpha", "standard"]
    assert [row["skill_name"] for row in plan["unmatched_artifacts"]] == ["unused"]
    assert not (tmp_path / "work/catalogs/generated/generated-only/unused").exists()
    assert not (tmp_path / "work/catalogs/generated/existing-plus-generated/unused").exists()


def test_combined_catalog_replaces_same_name_standard_skill(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["standard"])
    generated = tmp_path / "generated"
    _skill(generated, "standard")
    base = tmp_path / "base"
    (base / "org").mkdir(parents=True)
    (base / "org" / "standard.yaml").write_text(
        "name: standard\n"
        "title: Standard\n"
        "kind: recipe\n"
        "domain: org\n"
        "description: Base standard skill\n"
        "query: {schema: dm_core, logic_model: employee_actual, metrics: [fact_count]}\n",
        encoding="utf-8",
    )

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA,
        base_catalog_path=base,
        generated_modes=("existing_plus_generated",),
    )

    assert len(plan["groups"]) == 1
    group = plan["groups"][0]
    assert group["replaces_existing"] is True
    assert group["skill_names"] == ["standard"]
    combined = Path(group["combined_catalog_path"])
    assert not (combined / "org" / "standard.yaml").exists()
    registry = Registry.load(combined)
    assert registry.active() == ["standard"]
    assert registry.get("standard").kind == "reference"
    assert registry.get("standard").description == "Generated standard"
    assert registry.duplicates() == {}
