"""Content-addressed, additive snapshots of Heimdall skill catalogs."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .modes import _loaded_registry, _skill_files, catalog_hash


def _install(files: dict[Path, Path], snapshots_root: str | Path) -> Path:
    root = Path(snapshots_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".catalog-", dir=root) as work:
        staging = Path(work)
        for relative, source in sorted(files.items()):
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _loaded_registry(staging)
        digest = catalog_hash(staging)
        if not _skill_files(staging):
            raise ValueError("catalog snapshot cannot be empty")
        manifest = {
            "schema_version": "1.0",
            "catalog_hash": digest,
            "skill_files": sorted(path.as_posix() for path in files),
        }
        (staging / "snapshot-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        destination = root / digest.removeprefix("sha256:")
        if destination.exists():
            verify_snapshot(destination)
            if catalog_hash(destination) != digest:
                raise ValueError(f"catalog snapshot hash collision: {destination}")
            return destination
        staging.rename(destination)
        return destination


def verify_snapshot(path: str | Path) -> str:
    """Refuse changes to a previously pinned snapshot."""
    root = Path(path)
    manifest = json.loads((root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    actual_files = sorted(p.relative_to(root).as_posix() for p in _skill_files(root))
    actual_hash = catalog_hash(root)
    if actual_files != manifest["skill_files"] or actual_hash != manifest["catalog_hash"]:
        raise ValueError(f"catalog snapshot was modified: {root}")
    _loaded_registry(root)
    return actual_hash


def snapshot_catalog(source: str | Path, snapshots_root: str | Path) -> Path:
    """Create or reuse an immutable-by-hash copy; never edit the live catalog."""
    source = Path(source).resolve()
    _loaded_registry(source)
    files = {p.relative_to(source): p for p in _skill_files(source)}
    return _install(files, snapshots_root)


def compose_catalog(
    standard_snapshot: str | Path,
    generated_source: str | Path,
    snapshots_root: str | Path,
) -> Path:
    """Future non-mock mode: copy standard and add generated files atomically."""
    standard = Path(standard_snapshot).resolve()
    generated = Path(generated_source).resolve()
    verify_snapshot(standard)
    base_registry = _loaded_registry(standard)
    overlay_registry = _loaded_registry(generated)
    collisions = set(base_registry.all_names()) & set(overlay_registry.all_names())
    if collisions:
        raise ValueError(f"generated skill names collide with Heimdall: {sorted(collisions)}")
    files = {p.relative_to(standard): p for p in _skill_files(standard)}
    for path in _skill_files(generated):
        relative = path.relative_to(generated)
        if relative in files:
            raise ValueError(f"generated skill path collides with Heimdall: {relative}")
        files[relative] = path
    return _install(files, snapshots_root)
