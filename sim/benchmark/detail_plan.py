"""Matched experiments over Skill Factory method/detail catalogs; no LLM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from heimdall.skills.registry import Registry
from .cli import load_cases_path
from .generated_plan import _discover_artifacts, _variant_id, _workspace_path
from .modes import _loaded_registry
from .path_lib import SCHEMA_RELATIVE

LEVELS = ("abstract", "medium", "detailed")
METHODS = ("no-context", "compact", "full")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_detail_plan(
    cases_source: str | Path, generated_root: str | Path, output_root: str | Path,
    *, methods: tuple[str, ...] = ("no-context", "compact"),
    levels: tuple[str, ...] = LEVELS, limit: int | None = None, seed: int = 17,
    schema_path: str | Path = SCHEMA_RELATIVE, resume: bool = False,
    cases_per_skill: int | None = None,
) -> dict[str, Any]:
    """Create one catalog per method/detail, using identical fully covered cases.

    There must be exactly one physical artifact for each canonical skill in an
    arm. Multiple generations are a separate experimental factor and must not
    be selected silently. Authorial cases, prompts and skills remain untouched.
    """
    for selected, allowed, label in ((methods, METHODS, "methods"), (levels, LEVELS, "levels")):
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(allowed):
            raise ValueError(f"invalid {label}: {selected}")
    if len(levels) < 2:
        raise ValueError("a detail experiment needs at least two levels")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if cases_per_skill is not None and cases_per_skill < 1:
        raise ValueError("cases_per_skill must be >= 1")
    cases = load_cases_path(cases_source, schema_path=schema_path)
    source, root = Path(generated_root).resolve(), Path(output_root).resolve()
    registry = Registry.load(source)
    metadata = {}
    for item in registry.files():
        relative = item.path.relative_to(source)
        if item.path.name != "SKILL.md" or len(relative.parts) != 4:
            raise ValueError(f"expected <method>/<detail>/<artifact>/SKILL.md: {relative}")
        method, level = relative.parts[:2]
        if method not in METHODS or level not in LEVELS:
            raise ValueError(f"unknown method/detail in {relative}")
        metadata[_variant_id(item, source)] = (method, level)
    request = {
        "cases_source": str(Path(cases_source).resolve()), "generated_root": str(source),
        "methods": list(methods), "levels": list(levels), "limit": limit, "seed": seed,
        "cases_per_skill": cases_per_skill,
        "cases_hash": _digest([case.raw for case in cases]),
        "artifact_files_hash": _digest([
            (str(p.relative_to(source)), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(source.rglob("*")) if p.is_file()
        ]),
    }
    plan_path = root / "plan.json"
    if resume and not plan_path.is_file():
        raise ValueError("no recorded detail plan to resume")
    if resume and plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        existing.get("request", {}).setdefault("cases_per_skill", None)
        if existing.get("request") != request:
            raise ValueError("resume inputs differ from the recorded experiment")
        for group in existing["groups"]:
            _loaded_registry(Path(group["catalog_path"]))
        return existing
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"plan already exists: {root}; use --resume with identical inputs")
    root.mkdir(parents=True, exist_ok=True)
    artifacts = _discover_artifacts(source, root / "individual-catalogs", base_catalog_path=None)
    arms = {(method, level): {} for method in methods for level in levels}
    for artifact in artifacts:
        key = metadata[artifact.variant_id]
        if key not in arms:
            continue
        if artifact.expected_skill in arms[key]:
            raise ValueError(f"multiple generations for {key}/{artifact.expected_skill}")
        arms[key][artifact.expected_skill] = artifact
    empty_arms = [key for key, available in arms.items() if not available]
    if empty_arms:
        raise ValueError(f"empty method/detail catalogs: {empty_arms}")
    common_skills = set.intersection(*(set(available) for available in arms.values()))
    eligible, excluded = [], []
    for case in cases:
        expected = set(case.raw["expected_skills"])
        if expected and expected <= common_skills:
            eligible.append(case)
        else:
            missing = {
                f"{method}/{level}": sorted(expected - set(available))
                for (method, level), available in arms.items() if expected - set(available)
            }
            excluded.append({
                "case_id": case.case_id, "expected_skills": sorted(expected),
                "status": "detail_incomplete_coverage_excluded" if expected else "generated_no_expected_skill_excluded",
                "reason": "not covered in every requested arm" if expected else "no expected skill",
                "missing_by_arm": missing,
            })
    candidates = eligible
    if cases_per_skill is not None:
        priority = {"answerable": 0, "missing_skill": 1, "no_data": 2, "access_control": 3, "out_of_scope": 4}
        by_skill = {}
        for case in eligible:
            for name in case.raw["expected_skills"]:
                by_skill.setdefault(name, []).append(case)
        chosen = {case.case_id for rows in by_skill.values()
                  for case in sorted(rows, key=lambda item: (priority.get(item.raw["category"], 5), item.case_id))[:cases_per_skill]}
        candidates = [case for case in eligible if case.case_id in chosen]
    selected = candidates[:limit] if limit is not None else candidates
    if not selected:
        raise ValueError("no verified cases have matching skills in every arm")
    (root / "cases").mkdir()
    covered = root / "cases/covered.jsonl"
    _write_suite(covered, [case.raw for case in selected])
    groups = []
    for (method, level), available in arms.items():
        variant_id = f"{method}--{level}"
        catalog = root / "catalogs" / variant_id
        catalog.mkdir(parents=True)
        skill_map = {}
        for canonical, artifact in sorted(available.items()):
            shutil.copytree(Path(artifact.catalog_path) / artifact.skill_name, catalog / artifact.skill_name)
            skill_map[canonical] = artifact.skill_name
        loaded = _loaded_registry(catalog)
        if set(loaded.active()) != set(skill_map.values()):
            raise ValueError(f"catalog changed while preparing {variant_id}")
        suite = root / "cases" / f"{variant_id}.jsonl"
        _write_suite(suite, [
            {**case.raw, "expected_skills": [skill_map[name] for name in case.raw["expected_skills"]]}
            for case in selected
        ])
        groups.append({
            "variant_id": variant_id, "slug": variant_id, "generation_method": method,
            "detail_level": level, "skill_names": sorted(skill_map.values()),
            "expected_skills": sorted(skill_map), "skill_map": skill_map,
            "case_ids": [case.case_id for case in selected],
            "cases_path": _workspace_path(suite), "catalog_path": str(catalog),
            "combined_catalog_path": None,
            "artifacts": [
                {"canonical_skill": name, "skill_name": artifact.skill_name,
                 "variant_id": artifact.variant_id}
                for name, artifact in sorted(available.items())
            ],
        })
    random.Random(seed).shuffle(groups)
    plan = {
        "schema_version": "3.0", "experiment": "prompt_detail", "request": request,
        "cases_source": request["cases_source"], "generated_root": str(source),
        "source_case_ids": [case.case_id for case in cases],
        "eligible_case_ids": [case.case_id for case in eligible],
        "selected_case_ids": [case.case_id for case in selected],
        "case_skills": {case.case_id: case.raw["expected_skills"] for case in selected},
        "covered_cases_path": _workspace_path(covered), "generated_modes": ["generated_skills"],
        "groups": groups, "excluded": excluded, "partially_covered": [],
        "methods": list(methods), "detail_levels": list(levels), "phase_order_seed": seed,
        "common_skills": sorted(common_skills),
        "case_selection": {"strategy": "all_verified" if cases_per_skill is None else "balanced_per_skill", "cases_per_skill": cases_per_skill, "n_after_sampling": len(candidates)},
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def _write_suite(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--generated-skills", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", default=SCHEMA_RELATIVE)
    parser.add_argument("--methods", default="no-context,compact")
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases-per-skill", type=int)
    parser.add_argument("--resume", action="store_true")
    # Accept dispatcher arguments shared with the existing artifact planner.
    parser.add_argument("--base-skills")
    parser.add_argument("--modes", default="generated_skills", choices=["generated_skills"])
    try:
        args = parser.parse_args(argv)
        plan = build_detail_plan(
            args.cases, args.generated_skills, args.output, schema_path=args.schema,
            methods=tuple(args.methods.split(",")), levels=tuple(args.levels.split(",")),
            seed=args.seed, limit=args.limit, resume=args.resume, cases_per_skill=args.cases_per_skill,
        )
        print(json.dumps({"plan": str(Path(args.output).resolve() / "plan.json"),
                          "arms": len(plan["groups"]), "selected_cases": len(plan["selected_case_ids"]),
                          "excluded_cases": len(plan["excluded"])}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"detail planning failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
