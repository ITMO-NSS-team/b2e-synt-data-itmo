"""Benchmark checks that run before any LLM session is created."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from jsonschema import Draft202012Validator

from heimdall.catalog.model import Catalog
from heimdall.engine.compile import compile_query
from heimdall.skills.registry import Registry
from sim.emulator.identity import IdentityIndex
from sim.fingerprint import RunFingerprint

from .cases import BenchmarkCase, validate_case
from .catalog_snapshots import verify_snapshot
from .modes import DATA_TOOLS, SKILL_TOOLS, BenchmarkMode, ModeConfig, ModeConfigs, _loaded_registry

_JSON_FENCE = re.compile(r"[\x60]{3}(json|jsonc)\s*\n(.*?)\n[\x60]{3}", re.IGNORECASE | re.DOTALL)
_PINNED_VERSION = re.compile(r"^[^@\s]+@[1-9][0-9]*$")


@dataclass(frozen=True, slots=True)
class PreflightResult:
    case_id: str
    mode: str
    status: str  # ready | draft_skipped | mock_skipped
    fingerprint: RunFingerprint | None = None


def _check_query(body: dict, catalog: Catalog, label: str) -> None:
    if not isinstance(body, dict):
        raise ValueError(f"{label}: query must be an object")
    schema, logic_model = body.get("schema"), body.get("logic_model")
    model = catalog.get(schema, logic_model) if isinstance(schema, str) and isinstance(logic_model, str) else None
    if model is None or "v2" not in model.channels:
        raise ValueError(f"{label}: unknown or unavailable model {schema}.{logic_model}")
    try:
        compile_query(body, model)
    except Exception as exc:
        raise ValueError(f"{label}: invalid mcp_query: {exc}") from exc


def validate_skill_catalog(root: str | Path, model_catalog: Catalog) -> Registry:
    """Check loading, kind, declared models and all executable query examples."""
    root = Path(root)
    if (root / "snapshot-manifest.json").exists():
        verify_snapshot(root)
    registry = _loaded_registry(root)
    for name in registry.all_names():
        skill = registry.get(name)
        label = f"skill {name}"
        if skill.kind == "recipe":
            if skill.query is None:
                raise ValueError(f"{label}: recipe has no query")
            if skill.model is not None and (
                skill.model.get("schema"), skill.model.get("logic_model")
            ) != (skill.query.get("schema"), skill.query.get("logic_model")):
                raise ValueError(f"{label}: model and query point to different vitrines")
            _check_query(skill.query, model_catalog, label)
            for index, variant in enumerate(skill.variants):
                _check_query(variant.get("query"), model_catalog, f"{label} variant {index}")
        elif skill.kind == "reference":
            if not isinstance(skill.body, str) or not skill.body.strip():
                raise ValueError(f"{label}: reference body is empty")
            for index, (language, snippet) in enumerate(_JSON_FENCE.findall(skill.body)):
                try:
                    example = json.loads(snippet) if language.lower() == "json" else None
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{label}: invalid JSON example {index}: {exc}") from exc
                # jsonc snippets are illustrations and may contain placeholders.
                # A JSON filter fragment is documentation, not a full query.
                if isinstance(example, dict) and {"schema", "logic_model"} <= example.keys():
                    _check_query(example, model_catalog, f"{label} example {index}")
        else:
            raise ValueError(f"{label}: unsupported kind {skill.kind}")
    return registry


def _check_gold(case: BenchmarkCase) -> None:
    try:
        Draft202012Validator.check_schema(case.raw["gold_contract"])
        Draft202012Validator(case.raw["gold_contract"]).validate(case.raw["gold_answer"])
    except Exception as exc:
        raise ValueError(f"{case.case_id}: gold_answer violates gold_contract: {exc}") from exc


def preflight_case(
    case: BenchmarkCase,
    mode: ModeConfig,
    *,
    snapshot_root: str | Path,
    standard_catalog_path: str | Path,
    model_catalog_path: str | Path,
    agent_config_version: str,
    identity_factory: Callable[..., IdentityIndex] = IdentityIndex,
    schema_path: str | Path | None = None,
) -> PreflightResult:
    """Validate a case x mode. Draft and mock are non-errors but non-runnable."""
    validate_case(case.raw, schema_path=schema_path)
    if case.status == "draft":
        return PreflightResult(case.case_id, mode.name, "draft_skipped")
    case.require_ready()
    _check_gold(case)
    if mode.name not in {item.value for item in BenchmarkMode}:
        raise ValueError(f"unknown benchmark mode: {mode.name}")
    if mode.name == BenchmarkMode.SKILLS_DISABLED:
        if mode.skills_enabled or mode.tool_subset != DATA_TOOLS or mode.catalog_path is not None:
            raise ValueError("skills_disabled must deny find_skills/get_skill and have no catalog")
    elif not mode.skills_enabled or mode.tool_subset != SKILL_TOOLS:
        raise ValueError(f"{mode.name}: skill channel and tool subset must be enabled")
    if mode.name == BenchmarkMode.GENERATED_SKILL and mode.is_mock:
        return PreflightResult(case.case_id, mode.name, "mock_skipped")
    if not _PINNED_VERSION.fullmatch(agent_config_version):
        raise ValueError("agent_config_version must be pinned, e.g. benchmark_agent@1")
    if case.raw["snapshot_id"] != mode.common.snapshot_id:
        raise ValueError(f"{case.case_id}: snapshot_id differs from mode conditions")
    root = Path(snapshot_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or not (root / "truth" / "people.json").is_file():
        raise ValueError(f"data snapshot is unavailable: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_id") != case.raw["snapshot_id"]:
        raise ValueError(f"{case.case_id}: data snapshot manifest has a different snapshot_id")
    try:
        scope = identity_factory(root, hr_employee_ids=mode.common.hr_employee_ids).scope_for(
            case.raw["employee_id"]
        )
    except Exception as exc:
        raise ValueError(f"{case.case_id}: employee does not exist in snapshot: {exc}") from exc
    if scope.role != case.raw["employee_role"]:
        raise ValueError(
            f"{case.case_id}: employee role mismatch: expected {case.raw['employee_role']}, actual {scope.role}"
        )
    model_catalog = Catalog.load(model_catalog_path)
    standard = Path(standard_catalog_path)
    verify_snapshot(standard)
    validate_skill_catalog(standard, model_catalog)
    if mode.name != BenchmarkMode.SKILLS_DISABLED:
        if mode.catalog_path is None or mode.catalog_hash is None:
            raise ValueError(f"{mode.name}: skill catalog path/hash is missing")
        if verify_snapshot(mode.catalog_path) != mode.catalog_hash:
            raise ValueError(f"{mode.name}: configured catalog hash differs from snapshot")
        if Path(mode.catalog_path).resolve() != standard.resolve():
            registry = validate_skill_catalog(mode.catalog_path, model_catalog)
            if mode.name == BenchmarkMode.GENERATED_SKILL:
                missing = set(mode.generated_skill_names) - set(registry.active())
                if missing:
                    raise ValueError(f"generated target skills missing or inactive: {sorted(missing)}")
    fingerprint = RunFingerprint.create(
        agent_config_version=agent_config_version,
        prompt_registry_version=mode.common.prompt_registry_version,
        skill_registry_hash=case.raw["skill_registry_hash"],
        model_id=mode.common.model_id,
        temperature=mode.common.temperature,
        data_snapshot_hash=manifest["snapshot_id"],
        traps_enabled=mode.common.traps_enabled,
        latency_profile=mode.common.latency_profile,
        hr_employee_ids=mode.common.hr_employee_ids,
    )
    return PreflightResult(case.case_id, mode.name, "ready", fingerprint)


def preflight_suite(
    cases: Iterable[BenchmarkCase],
    modes: ModeConfigs,
    **kwargs: object,
) -> list[PreflightResult]:
    """Deterministic batch check; no LLM and no modification of authorial files."""
    return [
        preflight_case(case, mode, **kwargs)
        for case in cases
        for mode in modes
    ]
