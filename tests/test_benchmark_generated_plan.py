from __future__ import annotations

import json
from pathlib import Path

from heimdall.skills.registry import Registry

from sim.benchmark.generated_plan import _expected_skill, build_generated_plan
from tests.fixtures.constants import EXAMPLE, SCHEMA


def _skill(root: Path, name: str, *, variant_id: str | None = None) -> None:
    path = root / (variant_id or name) / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"title: {name}\n"
        "kind: reference\n"
        "domain: test\n"
        f"description: Generated {name}\n"
        "version: 0.1.0\n"
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


def test_limit_is_applied_after_generated_coverage_filter(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["missing"])
    _case(cases, "case-0102-00", ["alpha"])
    _case(cases, "case-0103-00", ["alpha"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha")

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA, limit=1,
    )

    assert plan["eligible_case_ids"] == ["case-0102-00", "case-0103-00"]
    assert plan["selected_case_ids"] == ["case-0102-00"]


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


def test_plan_never_bundles_multiple_expected_skills(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["beta", "alpha"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha-compact-a94e814444-b4d7da560f", variant_id="alpha_1")
    _skill(generated, "alpha-full-a94e814444-c4d7da560f", variant_id="alpha_2")
    _skill(generated, "beta-compact-b94e814444-d4d7da560f", variant_id="beta_1")
    _skill(generated, "beta-full-b94e814444-e4d7da560f", variant_id="beta_2")

    plan = build_generated_plan(
        cases, generated, tmp_path / "work", schema_path=SCHEMA,
    )

    assert plan["selected_case_ids"] == ["case-0101-00"]
    assert len(plan["groups"]) == 4
    assert all(len(group["skill_names"]) == 1 for group in plan["groups"])


def test_plan_builds_existing_plus_one_generated_catalog(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha"])
    generated = tmp_path / "generated"
    _skill(generated, "alpha", variant_id="alpha_1")
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
    assert Registry.load(group["catalog_path"]).active() == ["alpha"]
    assert Registry.load(group["combined_catalog_path"]).active() == ["alpha", "standard"]
