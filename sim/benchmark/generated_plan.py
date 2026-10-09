"""Plan case-scoped generated-skill benchmark phases.

The planner never calls an LLM and never mutates authorial cases or generated
artifacts. It creates one case suite shared by all requested baseline modes
and one isolated catalog/suite per generated artifact.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heimdall.skills.registry import Registry, SkillFile

from .cli import load_cases_path
from .modes import BenchmarkMode, _loaded_registry
from .path_lib import SCHEMA_RELATIVE

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_HASHED_SKILL = re.compile(
    r"^(?P<base>.+)-(?P<method>no-context|compact|full)-"
    r"(?P<prompt_hash>[0-9a-f]{10})(?:-(?P<generation_hash>[0-9a-f]{10}))?$"
)


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    """One physical artifact implementing a canonical generated skill."""

    skill_name: str
    expected_skill: str
    variant_id: str
    catalog_path: str
    combined_catalog_path: str | None


@dataclass(frozen=True, slots=True)
class GeneratedGroup:
    """One generated artifact and every case that targets it."""

    skill_names: tuple[str, ...]
    expected_skills: tuple[str, ...]
    variant_id: str
    slug: str
    case_ids: tuple[str, ...]
    cases_path: str
    catalog_path: str
    combined_catalog_path: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_names": list(self.skill_names),
            "expected_skills": list(self.expected_skills),
            "variant_id": self.variant_id,
            "slug": self.slug,
            "case_ids": list(self.case_ids),
            "cases_path": self.cases_path,
            "catalog_path": self.catalog_path,
            "combined_catalog_path": self.combined_catalog_path,
        }


def build_generated_plan(
    cases_source: str | Path,
    generated_root: str | Path,
    output_root: str | Path,
    *,
    schema_path: str | Path | None = None,
    limit: int | None = None,
    base_catalog_path: str | Path | None = None,
    generated_modes: tuple[str, ...] = (BenchmarkMode.GENERATED_SKILLS.value,),
) -> dict[str, Any]:
    """Create the covered baseline suite and isolated generated phases.

    A case is selected when at least one name in ``expected_skills`` has an
    active generated artifact. ``limit`` is applied after this coverage filter.
    All baseline modes run once over the selected suite. Every available skill
    variant gets an individual generated phase. When
    ``existing_plus_generated`` is requested, the plan also prepares a catalog
    containing the standard Heimdall skills plus that one generated artifact.
    """
    if limit is not None and limit < 1:
        raise ValueError("case limit must be >= 1")
    if not generated_modes or len(set(generated_modes)) != len(generated_modes):
        raise ValueError("generated modes must be non-empty and unique")
    allowed_modes = {
        BenchmarkMode.GENERATED_SKILLS.value,
        BenchmarkMode.EXISTING_PLUS_GENERATED.value,
    }
    unknown_modes = set(generated_modes) - allowed_modes
    if unknown_modes:
        raise ValueError(f"unsupported generated modes: {sorted(unknown_modes)}")
    if (
        BenchmarkMode.EXISTING_PLUS_GENERATED.value in generated_modes
        and base_catalog_path is None
    ):
        raise ValueError("existing_plus_generated requires base_catalog_path")
    cases = load_cases_path(cases_source, schema_path=schema_path)
    source = Path(generated_root).resolve()

    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"generated benchmark plan already exists: {root}")
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / "catalogs").mkdir(parents=True, exist_ok=True)

    artifacts = _discover_artifacts(
        source,
        root / "catalogs",
        base_catalog_path=(
            Path(base_catalog_path).resolve()
            if BenchmarkMode.EXISTING_PLUS_GENERATED.value in generated_modes
            else None
        ),
    )
    artifacts_by_name: dict[str, list[GeneratedArtifact]] = {}
    for artifact in artifacts:
        artifacts_by_name.setdefault(artifact.expected_skill, []).append(artifact)

    eligible = []
    excluded: list[dict[str, Any]] = []
    partially_covered: list[dict[str, Any]] = []
    for case in cases:
        expected = tuple(case.raw["expected_skills"])
        if not expected:
            excluded.append(_exclude(
                case.case_id, expected, "generated_no_expected_skill_excluded",
                "case has no expected skill",
            ))
        elif not (available := sorted(set(expected) & artifacts_by_name.keys())):
            excluded.append(_exclude(
                case.case_id, expected, "generated_skill_unavailable_excluded",
                f"generated skills are unavailable: {', '.join(sorted(expected))}",
            ))
        else:
            eligible.append(case)
            missing = sorted(set(expected) - set(available))
            if missing:
                partially_covered.append({
                    "case_id": case.case_id,
                    "available_skills": available,
                    "unavailable_skills": missing,
                })

    selected = eligible[:limit] if limit is not None else eligible
    if not selected:
        raise ValueError("no verified case is covered by an available generated skill")

    covered_path = root / "cases" / "covered.jsonl"
    _write_cases(covered_path, selected)

    cases_by_skill: dict[str, list[Any]] = {}
    for case in selected:
        for expected_skill in sorted(case.raw["expected_skills"]):
            if expected_skill in artifacts_by_name:
                cases_by_skill.setdefault(expected_skill, []).append(case)

    groups: list[GeneratedGroup] = []
    used_group_slugs: set[str] = set()
    for expected_skill, skill_cases in sorted(cases_by_skill.items()):
        for artifact in artifacts_by_name[expected_skill]:
            slug = _slug(artifact.variant_id, used_group_slugs)
            used_group_slugs.add(slug)
            cases_path = root / "cases" / f"{slug}.jsonl"
            _write_cases(cases_path, skill_cases, expected_skills=(artifact.skill_name,))
            groups.append(GeneratedGroup(
                skill_names=(artifact.skill_name,),
                expected_skills=(expected_skill,),
                variant_id=artifact.variant_id,
                slug=slug,
                case_ids=tuple(case.case_id for case in skill_cases),
                cases_path=_workspace_path(cases_path),
                catalog_path=artifact.catalog_path,
                combined_catalog_path=artifact.combined_catalog_path,
            ))

    payload = {
        "schema_version": "2.3",
        "cases_source": str(Path(cases_source).resolve()),
        "generated_root": str(source),
        "source_case_ids": [case.case_id for case in cases],
        "eligible_case_ids": [case.case_id for case in eligible],
        "selected_case_ids": [case.case_id for case in selected],
        "covered_cases_path": _workspace_path(covered_path),
        "generated_modes": list(generated_modes),
        "groups": [group.as_dict() for group in groups],
        "excluded": excluded,
        "partially_covered": partially_covered,
    }
    (root / "plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _discover_artifacts(
    source: Path,
    catalogs_root: Path,
    *,
    base_catalog_path: Path | None,
) -> list[GeneratedArtifact]:
    """Load physical variants while allowing repeated canonical skill names."""
    if base_catalog_path is not None:
        _loaded_registry(base_catalog_path)
    registry = Registry.load(source)
    errors = [item for item in registry.files() if not item.ok]
    if errors:
        details = "; ".join(f"{item.path}: {item.error}" for item in errors)
        raise ValueError(f"generated catalog contains invalid skills: {details}")

    files = [item for item in registry.files() if item.ok and item.name]
    if not files:
        raise ValueError(f"generated catalog contains no skill files: {source}")

    artifacts: list[GeneratedArtifact] = []
    used_variant_ids: set[str] = set()
    used_slugs: set[str] = set()
    for item in files:
        variant_id = _variant_id(item, source)
        if variant_id in used_variant_ids:
            raise ValueError(f"duplicate generated variant_id: {variant_id}")
        used_variant_ids.add(variant_id)
        slug = _slug(variant_id, used_slugs)
        used_slugs.add(slug)
        catalog = catalogs_root / "generated-only" / slug / str(item.name)
        catalog.mkdir(parents=True)
        _copy_skill_artifact(item.path, source, catalog)
        isolated = _loaded_registry(catalog.parent)
        if isolated.active() != [item.name]:
            raise ValueError(
                f"generated variant {variant_id} did not load as one active skill: "
                f"{item.name}"
            )
        combined_catalog_path = None
        if base_catalog_path is not None:
            combined_catalog_path = _combined_catalog(
                base_catalog_path,
                catalog.parent,
                catalogs_root / "existing-plus-generated" / slug,
                str(item.name),
            )
        artifacts.append(GeneratedArtifact(
            skill_name=str(item.name),
            expected_skill=_expected_skill(str(item.name)),
            variant_id=variant_id,
            catalog_path=str(catalog.parent),
            combined_catalog_path=combined_catalog_path,
        ))
    return sorted(artifacts, key=lambda item: (item.skill_name, item.variant_id))


def _variant_id(item: SkillFile, source: Path) -> str:
    path = item.path
    if path.name == "SKILL.md":
        relative_parent = path.parent.relative_to(source)
        raw = "--".join(relative_parent.parts) or str(item.name)
    else:
        raw = str(path.relative_to(source).with_suffix("")).replace("/", "--")
    variant_id = _SAFE.sub("-", raw).strip(".-")
    if not variant_id:
        raise ValueError(f"cannot derive generated variant_id from {path}")
    return variant_id


def _combined_catalog(
    base_catalog: Path,
    generated_catalog: Path,
    target: Path,
    generated_skill_name: str,
) -> str:
    """Copy the standard catalog and exactly one generated artifact."""
    shutil.copytree(base_catalog, target)
    for source in sorted(generated_catalog.rglob("*")):
        relative = source.relative_to(generated_catalog)
        destination = target / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if destination.exists():
            raise ValueError(f"combined catalog path collision: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    registry = _loaded_registry(target)
    if generated_skill_name not in registry.active():
        raise ValueError(
            f"combined catalog did not load generated skill: {generated_skill_name}"
        )
    return str(target)


def _exclude(case_id: str, expected: tuple[str, ...], status: str, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": status,
        "reason": reason,
        "expected_skills": list(expected),
    }


def _write_cases(
    path: Path,
    cases: list[Any],
    *,
    expected_skills: tuple[str, ...] | None = None,
) -> None:
    rows = []
    for case in cases:
        raw = dict(case.raw)
        if expected_skills is not None:
            raw["expected_skills"] = list(expected_skills)
        rows.append(raw)
    path.write_text(
        "".join(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n" for raw in rows),
        encoding="utf-8",
    )


def _expected_skill(skill_name: str) -> str:
    match = _HASHED_SKILL.fullmatch(skill_name)
    return match.group("base") if match else skill_name


def _copy_skill_artifact(source_file: Path, source_root: Path, target: Path) -> None:
    """Copy one skill and its colocated resources into an isolated catalog."""
    if source_file.name == "SKILL.md" and source_file.parent != source_root:
        shutil.copytree(source_file.parent, target, dirs_exist_ok=True)
        return
    shutil.copy2(source_file, target / source_file.name)


def _slug(value: str, used: set[str]) -> str:
    base = _SAFE.sub("-", value).strip(".-") or "skill"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _workspace_path(path: Path) -> str:
    """Return a path usable below the Compose runner's ``/app`` mount."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Plan isolated generated-skill phases")
    result.add_argument("--cases", required=True)
    result.add_argument("--generated-skills", required=True)
    result.add_argument("--base-skills")
    result.add_argument(
        "--modes",
        default=BenchmarkMode.GENERATED_SKILLS.value,
        help="comma-separated generated-skill modes",
    )
    result.add_argument("--output", required=True)
    result.add_argument("--schema", default=SCHEMA_RELATIVE)
    result.add_argument("--limit", type=int)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        plan = build_generated_plan(
            args.cases, args.generated_skills, args.output,
            schema_path=args.schema, limit=args.limit,
            base_catalog_path=args.base_skills,
            generated_modes=tuple(
                item.strip() for item in args.modes.split(",") if item.strip()
            ),
        )
        print(json.dumps({
            "plan": str(Path(args.output).resolve() / "plan.json"),
            "variants": len(plan["groups"]),
            "selected_cases": len(plan["selected_case_ids"]),
            "excluded_cases": len(plan["excluded"]),
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"generated benchmark planning failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
