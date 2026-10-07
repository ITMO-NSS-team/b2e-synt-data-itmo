from __future__ import annotations

import json
from pathlib import Path

from heimdall.skills.registry import Registry

from sim.benchmark.generated_plan import build_generated_plan
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


def _case(root: Path, case_id: str, expected: list[str]) -> None:
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

    assert plan["selected_case_ids"] == ["case-0101-00", "case-0101-01"]
    covered = [json.loads(line) for line in Path(plan["covered_cases_path"]).read_text().splitlines()]
    assert [row["case_id"] for row in covered] == ["case-0101-00", "case-0101-01"]
    assert len(plan["groups"]) == 2
    assert [group["variant_id"] for group in plan["groups"]] == ["alpha_1", "alpha_2"]
    for group in plan["groups"]:
        assert group["skill_name"] == "alpha"
        assert group["case_ids"] == ["case-0101-00", "case-0101-01"]
        isolated = Registry.load(group["catalog_path"])
        assert isolated.active() == ["alpha"]
        assert (Path(group["catalog_path"]) / "alpha" / "resource.txt").read_text() == (
            "supporting context"
        )
    assert [row["status"] for row in plan["excluded"]] == [
        "generated_skill_unavailable_excluded",
        "generated_no_expected_skill_excluded",
        "generated_multiple_skills_excluded",
    ]


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
