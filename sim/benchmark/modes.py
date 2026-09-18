"""Explicit mode configurations; no agent sessions or catalog writes here."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from heimdall.skills.registry import EXTENSIONS, Registry
from sim.fingerprint import VALID_LATENCY_PROFILES

DATA_TOOLS = ("list_models", "describe_model", "get_docs", "mcp_query")
SKILL_TOOLS = DATA_TOOLS + ("find_skills", "get_skill")
_PINNED_REF = re.compile(r"^[^@\s]+@[1-9][0-9]*$")


class BenchmarkMode(str, Enum):
    """Named experiment arms that share ``CommonConditions``.

    Attributes:
        SKILLS_DISABLED: Base information-service tools only; no skill catalog.
        EXISTING_SKILLS: Catalog of already-deployed skills plus ``find_skills``/``get_skill``.
        GENERATED_SKILLS: Combined catalog with generated skills; may still be a mock.
    """
    SKILLS_DISABLED = "skills_disabled"
    EXISTING_SKILLS = "existing_skills"
    GENERATED_SKILLS = "generated_skills"


@dataclass(frozen=True, slots=True)
class CommonConditions:
    """Shared experimental conditions across all benchmark modes.

    Attributes:
        model_id: Model identifier under test.
        temperature: Sampling temperature; must be finite.
        prompt_registry_version: Pinned prompt ref, e.g. ``system_prompt@1``.
        snapshot_id: Data corpus id that every case must match.
        traps_enabled: Whether the emulator injects trap records.
        latency_profile: Named emulator latency profile.
        hr_employee_ids: Extra HR-scope employee ids; stored sorted.
        code_execution: Must stay ``forbidden`` for this benchmark.
    """
    model_id: str
    temperature: float
    prompt_registry_version: str
    snapshot_id: str
    traps_enabled: bool
    latency_profile: str
    hr_employee_ids: tuple[str, ...]
    code_execution: str = "forbidden"

    def __post_init__(self) -> None:
        for field in ("model_id", "snapshot_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if not isinstance(self.prompt_registry_version, str) or not _PINNED_REF.fullmatch(
            self.prompt_registry_version
        ):
            raise ValueError("prompt_registry_version must be pinned, e.g. system_prompt@1")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and numeric")
        if not isinstance(self.traps_enabled, bool):
            raise ValueError("traps_enabled must be boolean")
        if self.latency_profile not in VALID_LATENCY_PROFILES:
            raise ValueError("unknown latency_profile")
        if self.code_execution != "forbidden":
            raise ValueError("benchmark modes require code_execution=forbidden")
        if not isinstance(self.hr_employee_ids, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.hr_employee_ids
        ):
            raise ValueError("hr_employee_ids must be a tuple of employee IDs")
        if len(set(self.hr_employee_ids)) != len(self.hr_employee_ids):
            raise ValueError("hr_employee_ids contains duplicates")
        object.__setattr__(self, "hr_employee_ids", tuple(sorted(self.hr_employee_ids)))


@dataclass(frozen=True, slots=True)
class ModeConfig:
    """Complete configuration of one named benchmark arm.

    Attributes:
        name: Mode name; must match ``BenchmarkMode``.
        tool_subset: Tools the agent may call in this arm.
        catalog_path: Snapshot path, or ``None`` when skills are disabled.
        catalog_hash: Content hash of that snapshot, or ``None``.
        generated_skills_path: Overlay catalog for generated skills, if any.
        generated_skills_hash: Content hash of the overlay, if any.
        generated_skill_names: Active generated skill names.
        common: Shared conditions copied into every arm.
        skills_enabled: Whether ``find_skills``/``get_skill`` are in the subset.
        is_mock: True when generated skills were not supplied.
    """
    name: str
    tool_subset: tuple[str, ...]
    catalog_path: str | None
    catalog_hash: str | None
    generated_skills_path: str | None
    generated_skills_hash: str | None
    generated_skill_names: tuple[str, ...]
    common: CommonConditions
    skills_enabled: bool = True
    is_mock: bool = False


@dataclass(frozen=True, slots=True)
class ModeConfigs:
    """The three standard arms of one experiment.

    Attributes:
        skills_disabled: Baseline without skills.
        existing_skills: Catalog of already-deployed skills.
        generated_skills: Combined catalog, possibly still a mock.
    """
    skills_disabled: ModeConfig
    existing_skills: ModeConfig
    generated_skills: ModeConfig

    def __post_init__(self) -> None:
        for mode, expected in (
            (self.skills_disabled, BenchmarkMode.SKILLS_DISABLED),
            (self.existing_skills, BenchmarkMode.EXISTING_SKILLS),
            (self.generated_skills, BenchmarkMode.GENERATED_SKILLS),
        ):
            if mode.name != expected.value:
                raise ValueError("mode key/name mismatch")

    def __iter__(self) -> Iterator[ModeConfig]:
        """Yield the three arms in fixed order: disabled, existing, generated."""
        yield self.skills_disabled
        yield self.existing_skills
        yield self.generated_skills


def _skill_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"skill catalog does not exist: {root}")
    return [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in EXTENSIONS and path.stem.upper() != "README"
    ]


def catalog_hash(root: str | Path) -> str:
    """Hash exact relative paths and bytes of skill files, not unrelated docs.

    Args:
        root: Skill catalog directory.

    Returns:
        Canonical ``sha256:<hex>`` digest of skill files in sorted order.
    """
    base = Path(root).resolve()
    digest = hashlib.sha256()
    for path in _skill_files(base):
        name = path.relative_to(base).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def _loaded_registry(root: Path) -> Registry:
    registry = Registry.load(root)
    errors = [f"{item.path}: {item.error}" for item in registry.files() if not item.ok]
    duplicates = registry.duplicates()
    if errors or duplicates:
        raise ValueError(f"invalid skill catalog: errors={errors}, duplicates={duplicates}")
    return registry


def build_modes(
    common: CommonConditions,
    base_catalog_path: str | Path,
    generated_skills_path: str | Path | None = None,
    *,
    snapshots_root: str | Path | None = None,
) -> ModeConfigs:
    """Pin a standard snapshot. Until generated skills exist, that mode is a mock.

    Args:
        common: Shared model, prompt, snapshot and emulator conditions.
        base_catalog_path: Live existing-skill catalog to snapshot.
        generated_skills_path: Optional overlay of generated skills.
        snapshots_root: Directory for content-addressed catalog copies.
            Required when ``generated_skills_path`` is set.

    Returns:
        Three mode configs sharing ``common``.

    Raises:
        ValueError: If a catalog is empty, invalid, or name-collides.
    """
    base = Path(base_catalog_path).resolve()
    if snapshots_root is not None:
        from .catalog_snapshots import snapshot_catalog
        base = snapshot_catalog(base, snapshots_root)
    standard = _loaded_registry(base)
    if not _skill_files(base):
        raise ValueError("standard catalog is empty")
    base_hash = catalog_hash(base)
    generated_path: str | None = None
    generated_hash: str | None = None
    combined_path = str(base)
    combined_hash = base_hash
    names: tuple[str, ...] = ()
    is_mock = generated_skills_path is None
    if generated_skills_path is not None:
        if snapshots_root is None:
            raise ValueError("generated skills require snapshots_root for a combined catalog")
        generated = Path(generated_skills_path).resolve()
        overlay = _loaded_registry(generated)
        if not _skill_files(generated):
            raise ValueError("generated_skills mode requires at least one generated skill")
        names = tuple(overlay.active())
        if len(names) != len(overlay.all_names()):
            raise ValueError("generated skills must be active")
        collisions = set(standard.all_names()) & set(overlay.all_names())
        if collisions:
            raise ValueError(f"generated skill names collide with existing catalog: {sorted(collisions)}")
        generated_path = str(generated)
        generated_hash = catalog_hash(generated)
        from .catalog_snapshots import compose_catalog
        combined = compose_catalog(base, generated, snapshots_root)
        combined_path = str(combined)
        combined_hash = catalog_hash(combined)
    return ModeConfigs(
        skills_disabled=ModeConfig(
            BenchmarkMode.SKILLS_DISABLED.value, DATA_TOOLS, None, None, None, None, (), common,
            skills_enabled=False),
        existing_skills=ModeConfig(
            BenchmarkMode.EXISTING_SKILLS.value, SKILL_TOOLS, str(base), base_hash, None, None, (), common),
        generated_skills=ModeConfig(
            BenchmarkMode.GENERATED_SKILLS.value, SKILL_TOOLS, combined_path, combined_hash,
            generated_path, generated_hash, names, common, is_mock=is_mock),
    )


def write_mode_config(modes: ModeConfigs, output: str | Path) -> Path:
    """Persist all experiment variables explicitly for subsequent preflight.

    Args:
        modes: The three arms; they must share identical ``common`` conditions.
        output: Destination JSON path.

    Returns:
        Path written.

    Raises:
        ValueError: If tool subsets, skill flags or generated-mode fields drift.
    """
    if len({mode.common for mode in modes}) != 1:
        raise ValueError("all modes must share identical common conditions")
    if modes.skills_disabled.tool_subset != DATA_TOOLS or any(
        mode.tool_subset != SKILL_TOOLS
        for mode in (modes.existing_skills, modes.generated_skills)
    ):
        raise ValueError("mode tool subsets do not match benchmark contract")
    if (
        modes.skills_disabled.skills_enabled
        or modes.existing_skills.skills_enabled is not True
        or modes.generated_skills.skills_enabled is not True
    ):
        raise ValueError("skill channel flags do not match benchmark modes")
    generated = modes.generated_skills
    if generated.is_mock and (generated.generated_skills_path is not None or generated.generated_skill_names):
        raise ValueError("mock generated mode must not contain generated skills")
    if not generated.is_mock and not generated.generated_skill_names:
        raise ValueError("generated mode requires target skills")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "modes": {mode.name: asdict(mode) for mode in sorted(modes, key=lambda item: item.name)},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
