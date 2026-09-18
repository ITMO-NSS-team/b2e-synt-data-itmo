"""Capture the conditions that are actually active in the Compose stand.

This module is executed inside ``admin-ui``: that container owns the writable
registry and can reach the emulator on the internal network.  The resulting
JSON is consumed by the host-side benchmark driver; secrets are never included.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

from sim.registry import Registry
from sim.skill_eval.pin import pin
from sim.skills import SkillStore

from .env import load_env, require_env
from .modes import BenchmarkMode


def pin_live_configs(
    registry: Registry, *, model_id: str | None = None,
    base_pinner: Callable[[Registry], dict[str, str]] = pin,
) -> dict[str, str]:
    """Pin the benchmark arms, optionally overriding only their model.

    Args:
        registry: Writable agent/prompt/skill registry.
        model_id: If set, rewrite ``model_id`` on each pinned arm.
        base_pinner: Creates skills-on/off configs; defaults to ``skill_eval.pin``.

    Returns:
        Mapping ``skills_on`` / ``skills_off`` to pinned ``name@N`` refs.
    """
    refs = base_pinner(registry)
    selected_model = (model_id or "").strip()
    if not selected_model:
        return refs
    updated: dict[str, str] = {}
    for key, ref in refs.items():
        version, raw = registry.load(ref)
        body = dict(raw)
        body["model_id"] = selected_model
        committed = registry.commit(
            version.name,
            "agent",
            body,
            actor="benchmark_smoke",
            note=f"benchmark: model override {selected_model}",
        )
        updated[key] = committed.ref
    return updated


def capture_live_config(
    registry: Registry,
    emulator_config: dict[str, Any],
    *,
    pinner: Callable[[Registry], dict[str, str]] = pin,
) -> dict[str, Any]:
    """Pin skills-on/off configs and return a secret-free experiment manifest.

    Args:
        registry: Writable registry inside admin-ui.
        emulator_config: ``/control/config`` payload from the emulator.
        pinner: Returns skills-on/off refs; defaults to ``pin``.

    Returns:
        Manifest with refs, config bodies, prompt versions, skill hash and
        emulator condition. Secrets are never included.

    Raises:
        ValueError: If a pinned config is malformed or emulator fields are missing.
    """
    pinned = pinner(registry)
    refs = {
        BenchmarkMode.SKILLS_DISABLED.value: pinned["skills_off"],
        BenchmarkMode.EXISTING_SKILLS.value: pinned["skills_on"],
    }
    configs: dict[str, dict[str, Any]] = {}
    prompt_versions: dict[str, str] = {}
    for ref in refs.values():
        _version, raw = registry.load(ref)
        if not isinstance(raw, dict) or not isinstance(raw.get("system_prompt_ref"), str):
            raise ValueError(f"pinned agent config is malformed: {ref}")
        prompt_version, _prompt = registry.load(raw["system_prompt_ref"])
        configs[ref] = raw
        prompt_versions[ref] = prompt_version.ref

    required = {
        "data_snapshot_hash", "traps_enabled", "latency_profile",
        "hr_employee_ids",
    }
    missing = required - emulator_config.keys()
    if missing:
        raise ValueError(f"emulator condition is incomplete: {sorted(missing)}")
    condition = {key: emulator_config[key] for key in sorted(required)}
    return {
        "schema_version": "1.0",
        "refs": refs,
        "configs": configs,
        "prompt_versions": prompt_versions,
        "skill_registry_hash": SkillStore(registry).registry_hash(),
        "emulator": condition,
    }


def fetch_json(url: str, *, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Read one internal JSON endpoint; dependency injection keeps tests offline.

    Args:
        url: Absolute URL, typically the emulator control config.
        opener: Optional ``urlopen``-compatible callable.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the body is not a JSON object.
    """
    if opener is None:
        from urllib.request import urlopen

        opener = urlopen
    with opener(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def main() -> None:
    """Print the live-stand manifest as JSON to stdout.

    Reads ``B2E_REGISTRY_DB``, ``HEIMDALL_URL`` and optional ``B2E_BENCH_MODEL``
    from local ``deploy/.env``, overlaid by the process environment.
    """
    load_env()
    registry_path = require_env("B2E_REGISTRY_DB")
    emulator_url = require_env("HEIMDALL_URL")
    registry = Registry(registry_path)
    try:
        payload = capture_live_config(
            registry,
            fetch_json(f"{emulator_url.rstrip('/')}/control/config"),
            pinner=lambda target: pin_live_configs(
                target,
                model_id=(os.getenv("B2E_BENCH_MODEL") or "").strip() or None,
            ),
        )
    finally:
        registry.close()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
