"""Plan case-scoped generated-skill benchmark phases.

The planner never calls an LLM and never mutates authorial cases or generated
artifacts. It creates one case suite shared by baseline modes, one isolated
suite per generated artifact, and one full-catalog suite per source when asked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from heimdall.skills.registry import BENCHMARK_GENERATED_PREFIX, Registry, SkillFile, parse_text

from .cli import load_cases_path
from .env import load_env
from .modes import BenchmarkMode, _loaded_registry, catalog_hash
from .path_lib import SCHEMA_RELATIVE

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_SOURCE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_VERSION_SUFFIX = re.compile(r"-v[0-9]+-[0-9]+-[0-9]+$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_HASHED_SKILL = re.compile(
    r"^(?P<base>.+)-(?P<method>no-context|compact|full)-"
    r"(?P<prompt_hash>[0-9a-f]{10})(?:-(?P<generation_hash>[0-9a-f]{10}))?$"
)
# shortcut: SF's trailing 10-hex hash is naming convention; use explicit logical metadata if names can end that way.
_CONTENT_HASHED_SKILL = re.compile(r"^(?P<base>.+)-[0-9a-f]{10}$")
_VERSIONED_SKILL = re.compile(r"(?P<logical>.+)-v[0-9]+-[0-9]+-[0-9]+$")


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    """One physical artifact implementing a canonical generated skill."""

    skill_name: str
    expected_skills: tuple[str, ...]
    source: str
    source_path: str
    path_labels: tuple[str, ...]
    version: str
    variant_id: str
    catalog_path: str
    combined_catalog_path: str | None
    shares_name_with_existing: bool


@dataclass(frozen=True, slots=True)
class GeneratedGroup:
    """One generated artifact and every case that targets it."""

    skill_names: tuple[str, ...]
    expected_skills: tuple[str, ...]
    variant_id: str
    modes: tuple[str, ...]
    scope: str
    source: str
    source_path: str
    path_labels: tuple[str, ...]
    version: str
    slug: str
    case_ids: tuple[str, ...]
    matched_case_ids: tuple[str, ...]
    cases_path: str
    catalog_path: str
    combined_catalog_path: str | None
    shares_name_with_existing: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_names": list(self.skill_names),
            "expected_skills": list(self.expected_skills),
            "variant_id": self.variant_id,
            "modes": list(self.modes),
            "scope": self.scope,
            "source": self.source,
            "source_path": self.source_path,
            "path_labels": list(self.path_labels),
            "version": self.version,
            "slug": self.slug,
            "case_ids": list(self.case_ids),
            "matched_case_ids": list(self.matched_case_ids),
            "cases_path": self.cases_path,
            "catalog_path": self.catalog_path,
            "combined_catalog_path": self.combined_catalog_path,
            "shares_name_with_existing": self.shares_name_with_existing,
        }


def build_generated_plan(
    cases_source: str | Path,
    generated_root: str | Path | None,
    output_root: str | Path,
    *,
    schema_path: str | Path | None = None,
    limit: int | None = None,
    base_catalog_path: str | Path | None = None,
    generated_modes: tuple[str, ...] = (BenchmarkMode.GENERATED_SKILLS.value,),
    generated_sources: Mapping[str, str | Path] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Create the covered baseline suite and isolated generated phases.

    For individual-skill modes, a case is selected when one expected skill has
    an artifact. The catalog mode selects all input cases. ``limit`` follows
    that selection.
    All baseline modes run once over the selected suite. Every available skill
    variant gets an individual generated phase. When
    ``existing_plus_generated`` is requested, the plan also prepares a catalog
    containing the standard Heimdall skills plus that one generated artifact.
    ``existing_plus_catalog`` adds one phase per source with all its skills.
    The generated artifact remains available alongside the standard skills,
    including another version of a skill with the same logical name.
    """
    if limit is not None and limit < 1:
        raise ValueError("case limit must be >= 1")
    if not generated_modes or len(set(generated_modes)) != len(generated_modes):
        raise ValueError("generated modes must be non-empty and unique")
    allowed_modes = {
        BenchmarkMode.GENERATED_SKILLS.value,
        BenchmarkMode.EXISTING_PLUS_GENERATED.value,
        BenchmarkMode.EXISTING_PLUS_CATALOG.value,
    }
    unknown_modes = set(generated_modes) - allowed_modes
    if unknown_modes:
        raise ValueError(f"unsupported generated modes: {sorted(unknown_modes)}")
    if (
        set(generated_modes) & {
            BenchmarkMode.EXISTING_PLUS_GENERATED.value,
            BenchmarkMode.EXISTING_PLUS_CATALOG.value,
        }
        and base_catalog_path is None
    ):
        raise ValueError("combined generated modes require base_catalog_path")
    cases = load_cases_path(cases_source, schema_path=schema_path)
    sources = (
        {label: Path(path).resolve() for label, path in generated_sources.items()}
        if generated_sources is not None else
        {"generated": Path(generated_root).resolve()} if generated_root is not None else {}
    )
    if not sources:
        raise ValueError("at least one generated skill catalog is required")
    for label, source in sources.items():
        if not _SOURCE_LABEL.fullmatch(label):
            raise ValueError(f"invalid generated skill source label: {label}")
        if not source.is_dir():
            raise ValueError(f"generated skill catalog is not a directory: {source}")

    root = Path(output_root).resolve()
    request = {
        "cases_hash": hashlib.sha256(json.dumps(
            [case.raw for case in cases], ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest(),
        "sources": {
            label: {"path": str(source), "hash": _source_hash(source)}
            for label, source in sorted(sources.items())
        },
        "base_catalog_hash": (
            catalog_hash(base_catalog_path) if base_catalog_path is not None else None
        ),
        "generated_modes": list(generated_modes),
        "limit": limit,
    }
    plan_path = root / "plan.json"
    if resume and plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("request") != request:
            raise ValueError("resume inputs differ from the recorded generated plan")
        return existing
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"generated benchmark plan already exists: {root}")

    catalog_mode = BenchmarkMode.EXISTING_PLUS_CATALOG.value
    individual_modes = tuple(mode for mode in generated_modes if mode != catalog_mode)
    artifacts = []
    named_sources = generated_sources is not None
    for label, source in sorted(sources.items()):
        artifacts.extend(_discover_artifacts(
            source,
            root / "catalogs" / label,
            source_label=label,
            variant_prefix=f"{label}--" if named_sources else "",
            base_catalog_path=(
                Path(base_catalog_path).resolve()
                if set(generated_modes) & {
                    BenchmarkMode.EXISTING_PLUS_GENERATED.value, catalog_mode,
                }
                else None
            ),
        ))
    artifacts_by_name: dict[str, list[GeneratedArtifact]] = {}
    for artifact in artifacts:
        for expected_skill in artifact.expected_skills:
            artifacts_by_name.setdefault(expected_skill, []).append(artifact)

    eligible = []
    excluded: list[dict[str, Any]] = []
    partially_covered: list[dict[str, Any]] = []
    for case in cases:
        if catalog_mode in generated_modes:
            eligible.append(case)
            continue
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
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / "catalogs").mkdir(parents=True, exist_ok=True)
    covered_path = root / "cases" / "covered.jsonl"
    _write_cases(covered_path, selected)

    groups: list[GeneratedGroup] = []
    used_group_slugs: set[str] = set()
    unmatched_artifacts = []
    for artifact in sorted(artifacts, key=lambda item: item.variant_id):
        skill_cases = [case for case in selected
                       if set(case.raw["expected_skills"]) & set(artifact.expected_skills)]
        if not skill_cases:
            unmatched_artifacts.append({
                "source": artifact.source, "variant_id": artifact.variant_id,
                "skill_name": artifact.skill_name, "version": artifact.version,
                "expected_skills": list(artifact.expected_skills),
            })
            continue
        if not individual_modes:
            continue
        catalog = Path(artifact.catalog_path)
        skill_dir = catalog / artifact.skill_name
        skill_dir.mkdir(parents=True)
        _copy_skill_artifact(Path(artifact.source_path), sources[artifact.source], skill_dir)
        isolated = _loaded_registry(catalog)
        if isolated.active() != [artifact.skill_name]:
            raise ValueError(
                f"generated variant {artifact.variant_id} did not load as one active skill: "
                f"{artifact.skill_name}"
            )
        runtime_names = ()
        if artifact.combined_catalog_path is not None:
            runtime_names = _combined_catalog(
                Path(base_catalog_path).resolve(), catalog,
                Path(artifact.combined_catalog_path), artifact.skill_name,
                artifact.version,
            )
        slug = _slug(artifact.variant_id, used_group_slugs)
        used_group_slugs.add(slug)
        cases_path = root / "cases" / f"{slug}.jsonl"
        matched_expected = tuple(sorted({name for case in skill_cases
                                         for name in case.raw["expected_skills"]
                                         if name in artifact.expected_skills}))
        _write_cases(cases_path, skill_cases, expected_skills=tuple(dict.fromkeys(
            (*matched_expected, artifact.skill_name, *runtime_names)
        )))
        groups.append(GeneratedGroup(
            skill_names=(artifact.skill_name,),
            expected_skills=matched_expected,
            variant_id=artifact.variant_id,
            modes=individual_modes,
            scope="skill",
            source=artifact.source,
            source_path=artifact.source_path,
            path_labels=artifact.path_labels,
            version=artifact.version,
            slug=slug,
            case_ids=tuple(case.case_id for case in skill_cases),
            matched_case_ids=tuple(case.case_id for case in skill_cases),
            cases_path=_workspace_path(cases_path),
            catalog_path=artifact.catalog_path,
            combined_catalog_path=artifact.combined_catalog_path,
            shares_name_with_existing=artifact.shares_name_with_existing,
        ))

    if catalog_mode in generated_modes and selected:
        for label, source in sorted(sources.items()):
            variant_id = f"catalog--{label}"
            if variant_id in {artifact.variant_id for artifact in artifacts}:
                raise ValueError(f"catalog variant_id conflicts with a skill: {variant_id}")
            slug = _slug(variant_id, used_group_slugs)
            used_group_slugs.add(slug)
            combined = root / "catalogs" / label / "existing-plus-catalog"
            runtime_names = _combined_catalog(
                Path(base_catalog_path).resolve(), source, combined,
            )
            registry = _loaded_registry(combined)
            targets = {
                name: set(_expected_names(registry.get(name).name, registry.get(name).version))
                for name in runtime_names
            }
            available_expected = set().union(*targets.values())
            case_targets = {
                case.case_id: tuple(name for name, expected in targets.items()
                                    if expected & set(case.raw["expected_skills"]))
                for case in selected
            }
            cases_path = root / "cases" / f"{slug}.jsonl"
            _write_cases(cases_path, selected, runtime_names_by_case=case_targets)
            matched = tuple(case_id for case_id, names in case_targets.items() if names)
            groups.append(GeneratedGroup(
                skill_names=tuple(registry.get(name).name for name in runtime_names),
                expected_skills=tuple(sorted({name for case in selected
                                              for name in case.raw["expected_skills"]}
                                             & available_expected)),
                variant_id=variant_id,
                modes=(catalog_mode,),
                scope="catalog",
                source=label,
                source_path=str(source),
                path_labels=(),
                version="",
                slug=slug,
                case_ids=tuple(case.case_id for case in selected),
                matched_case_ids=matched,
                cases_path=_workspace_path(cases_path),
                catalog_path=str(combined),
                combined_catalog_path=str(combined),
                shares_name_with_existing=any(artifact.shares_name_with_existing
                                              for artifact in artifacts if artifact.source == label),
            ))

    payload = {
        "schema_version": "2.3",
        "cases_source": str(Path(cases_source).resolve()),
        "generated_root": str(next(iter(sources.values()))) if len(sources) == 1 else None,
        "sources": {label: str(source) for label, source in sorted(sources.items())},
        "request": request,
        "source_case_ids": [case.case_id for case in cases],
        "eligible_case_ids": [case.case_id for case in eligible],
        "selected_case_ids": [case.case_id for case in selected],
        "covered_cases_path": _workspace_path(covered_path),
        "generated_modes": list(generated_modes),
        "groups": [group.as_dict() for group in groups],
        "excluded": excluded,
        "partially_covered": partially_covered,
        "unmatched_artifacts": unmatched_artifacts,
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
    source_label: str = "generated",
    variant_prefix: str = "",
) -> list[GeneratedArtifact]:
    """Load physical variants while allowing repeated canonical skill names."""
    base_registry = (
        _loaded_registry(base_catalog_path) if base_catalog_path is not None else None
    )
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
    base_names = (
        {base_registry.get(name).name for name in base_registry.all_names()}
        if base_registry is not None else set()
    )
    for item in files:
        variant_id = variant_prefix + _variant_id(item, source)
        if variant_id in used_variant_ids:
            raise ValueError(f"duplicate generated variant_id: {variant_id}")
        used_variant_ids.add(variant_id)
        slug = _slug(variant_id, used_slugs)
        used_slugs.add(slug)
        skill = parse_text(item.path.read_text(encoding="utf-8"), item.path)
        if skill.status != "active":
            raise ValueError(
                f"generated variant {variant_id} did not load as one active skill: "
                f"{skill.name}"
            )
        artifacts.append(GeneratedArtifact(
            skill_name=skill.name,
            expected_skills=_expected_names(skill.name, skill.version),
            source=source_label,
            source_path=str(item.path),
            path_labels=tuple(item.path.relative_to(source).parts[:-2]),
            version=skill.version,
            variant_id=variant_id,
            catalog_path=str(catalogs_root / "generated-only" / slug),
            combined_catalog_path=(
                str(catalogs_root / "existing-plus-generated" / slug)
                if base_catalog_path is not None else None
            ),
            shares_name_with_existing=skill.name in base_names,
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
    generated_skill_name: str | None = None,
    generated_version: str | None = None,
) -> tuple[str, ...]:
    """Copy standard skills and generated skills, preserving their metadata."""
    shutil.copytree(base_catalog, target)
    index = 1
    while (target / f"{BENCHMARK_GENERATED_PREFIX}{index}").exists():
        index += 1
    generated_copy = target / f"{BENCHMARK_GENERATED_PREFIX}{index}"
    shutil.copytree(generated_catalog, generated_copy)
    registry = _loaded_registry(target)
    if _pin_related_in_copy(registry, generated_copy):
        registry = _loaded_registry(target)
    generated_names = tuple(item.name for item in registry.files()
                            if item.ok and item.name and item.path.is_relative_to(generated_copy))
    if not generated_names:
        raise ValueError(
            f"combined catalog did not load generated skill: {generated_skill_name}"
        )
    if generated_skill_name is None:
        return generated_names
    return tuple(name for name in registry.active()
                 if registry.get(name).name == generated_skill_name
                 and registry.get(name).version == generated_version)


def _pin_related_in_copy(registry: Registry, generated_copy: Path) -> bool:
    """Keep old related links usable when generated skills duplicate their names."""
    names = set(registry.all_names())
    base_keys = {}
    for key in sorted(names):
        skill = registry.get(key)
        if skill.path is not None and not skill.path.is_relative_to(generated_copy):
            base_keys[skill.name] = key
            base_keys[f"{skill.name}@{skill.version}"] = key

    changed = False
    for item in registry.files():
        if not item.ok or not item.name:
            continue
        skill = registry.get(item.name)
        replacements = {link["name"]: base_keys[link["name"]]
                        for link in skill.related
                        if link.get("name") in base_keys and link["name"] not in names}
        if not replacements:
            continue
        text = item.path.read_text(encoding="utf-8")
        if item.path.suffix == ".md":
            end = text.index("\n---", 3)
            meta = yaml.safe_load(text[3:end])
            body = text[end + 4:].lstrip("\n")
        else:
            meta = yaml.safe_load(text)
        for link in meta["related"]:
            if link.get("name") in replacements:
                link["name"] = replacements[link["name"]]
        header = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
        item.path.write_text(
            f"---\n{header}---\n\n{body}" if item.path.suffix == ".md" else header,
            encoding="utf-8",
        )
        changed = True
    return changed


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
    runtime_names_by_case: Mapping[str, tuple[str, ...]] | None = None,
) -> None:
    rows = []
    for case in cases:
        raw = dict(case.raw)
        if expected_skills is not None:
            raw["expected_skills"] = list(expected_skills)
        if runtime_names_by_case is not None:
            raw["expected_skills"] = list(dict.fromkeys((
                *raw["expected_skills"], *runtime_names_by_case[case.case_id],
            )))
        rows.append(raw)
    path.write_text(
        "".join(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n" for raw in rows),
        encoding="utf-8",
    )


def _expected_skill(skill_name: str) -> str:
    match = _HASHED_SKILL.fullmatch(skill_name) or _CONTENT_HASHED_SKILL.fullmatch(skill_name)
    return match.group("base") if match else skill_name


def _expected_names(skill_name: str, version: str) -> tuple[str, ...]:
    name = _expected_skill(skill_name)
    base = _VERSION_SUFFIX.sub("", name)
    if _VERSION.fullmatch(version) and version != "0.0.0":
        versioned = f"{base}-v{version.replace('.', '-')}"
        return (versioned,) if base != name else (name, versioned)
    return (name,) if base == name else ()


def _source_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _configured_sources(raw: str) -> dict[str, Path]:
    sources = {}
    for entry in raw.split(","):
        label, separator, path = entry.strip().partition("=")
        if not separator or not label or not path.strip() or label in sources:
            raise ValueError(f"invalid BENCH_SKILL_CATALOGS entry: {entry}")
        sources[label] = Path(path.strip())
    return sources


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
    result.add_argument("--generated-skills")
    result.add_argument("--base-skills")
    result.add_argument(
        "--modes",
        default=BenchmarkMode.GENERATED_SKILLS.value,
        help="comma-separated generated-skill modes",
    )
    result.add_argument("--output", required=True)
    result.add_argument("--schema", default=SCHEMA_RELATIVE)
    result.add_argument("--limit", type=int)
    result.add_argument("--resume", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        load_env()
        args = parser().parse_args(argv)
        catalogs = os.getenv("BENCH_SKILL_CATALOGS", "").strip()
        plan = build_generated_plan(
            args.cases, args.generated_skills, args.output,
            schema_path=args.schema, limit=args.limit,
            base_catalog_path=args.base_skills,
            generated_sources=_configured_sources(catalogs) if catalogs else None,
            generated_modes=tuple(
                item.strip() for item in args.modes.split(",") if item.strip()
            ),
            resume=args.resume,
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
