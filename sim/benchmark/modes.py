"""Explicit mode configurations; no agent sessions or catalog writes here."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from heimdall.skills.registry import EXTENSIONS, Registry
from sim.fingerprint import VALID_LATENCY_PROFILES

DATA_TOOLS = ("list_models", "describe_model", "get_docs", "mcp_query")
SKILL_TOOLS = DATA_TOOLS + ("find_skills", "get_skill")
_PINNED_REF = re.compile(r"^[^@\s]+@[1-9][0-9]*$")


class BenchmarkMode(str, Enum):
    SKILLS_DISABLED = "skills_disabled"
    HEIMDALL_SKILLS = "heimdall_skills"
    GENERATED_SKILL = "generated_skill"


@dataclass(frozen=True, slots=True)
class CommonConditions:
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
    name: str
    tool_subset: tuple[str, ...]
    catalog_path: str | None
    catalog_hash: str | None
    generated_skills_path: str | None
    generated_skills_hash: str | None
    generated_skill_names: tuple[str, ...]
    common: CommonConditions

    def as_dict(self) -> dict:
        return asdict(self)


def _skill_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"skill catalog does not exist: {root}")
    return [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in EXTENSIONS and path.stem.upper() != "README"
    ]


def catalog_hash(root: str | Path) -> str:
    """Hash exact relative paths and bytes of skill files, not unrelated docs."""
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
    generated_skills_path: str | Path,
) -> dict[str, ModeConfig]:
    """Pin Heimdall's standard catalog and the additive generated overlay."""
    base = Path(base_catalog_path).resolve()
    generated = Path(generated_skills_path).resolve()
    standard = _loaded_registry(base)
    overlay = _loaded_registry(generated)
    if not _skill_files(base):
        raise ValueError("standard catalog is empty")
    if not _skill_files(generated):
        raise ValueError("generated_skill mode requires at least one generated skill")
    names = tuple(overlay.active())
    if len(names) != len(overlay.all_names()):
        raise ValueError("generated skills must be active")
    collisions = set(standard.all_names()) & set(overlay.all_names())
    if collisions:
        raise ValueError(f"generated skill names collide with Heimdall: {sorted(collisions)}")
    base_hash = catalog_hash(base)
    generated_hash = catalog_hash(generated)
    combined = hashlib.sha256((base_hash + generated_hash).encode("ascii")).hexdigest()
    return {
        BenchmarkMode.SKILLS_DISABLED.value: ModeConfig(
            BenchmarkMode.SKILLS_DISABLED.value, DATA_TOOLS, None, None, None, None, (), common),
        BenchmarkMode.HEIMDALL_SKILLS.value: ModeConfig(
            BenchmarkMode.HEIMDALL_SKILLS.value, SKILL_TOOLS, str(base), base_hash, None, None, (), common),
        BenchmarkMode.GENERATED_SKILL.value: ModeConfig(
            BenchmarkMode.GENERATED_SKILL.value, SKILL_TOOLS, str(base), "sha256:" + combined,
            str(generated), generated_hash, names, common),
    }


def write_mode_config(modes: dict[str, ModeConfig], output: str | Path) -> Path:
    """Persist all experiment variables explicitly for subsequent preflight."""
    expected = {mode.value for mode in BenchmarkMode}
    if set(modes) != expected:
        raise ValueError(f"mode config requires exactly {sorted(expected)}")
    if any(key != mode.name for key, mode in modes.items()):
        raise ValueError("mode key/name mismatch")
    if len({mode.common for mode in modes.values()}) != 1:
        raise ValueError("all modes must share identical common conditions")
    disabled = modes[BenchmarkMode.SKILLS_DISABLED]
    standard = modes[BenchmarkMode.HEIMDALL_SKILLS]
    generated = modes[BenchmarkMode.GENERATED_SKILL]
    if disabled.tool_subset != DATA_TOOLS or any(
        mode.tool_subset != SKILL_TOOLS for mode in (standard, generated)
    ):
        raise ValueError("mode tool subsets do not match benchmark contract")
    if generated.catalog_path != standard.catalog_path:
        raise ValueError("generated mode must extend the standard Heimdall catalog")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "modes": {key: modes[key].as_dict() for key in sorted(modes)}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
