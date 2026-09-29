"""Capture the conditions that are actually active in the Compose stand.

This module is executed inside ``admin-ui``: that container owns the writable
registry and can reach the emulator on the internal network.  The resulting
JSON is consumed by the benchmark runner; secrets are never included.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from sim.agent.provider import runtime_agent_config
from sim.registry import Registry
from .pin import pin
from sim.skills import SkillStore

from .env import load_env, require_env
from .modes import BenchmarkMode


def pin_live_configs(
    registry: Registry,
    base_pinner: Callable[[Registry], dict[str, str]] = pin,
) -> dict[str, str]:
    """Pin the benchmark arms.

    Args:
        registry: Writable agent/prompt/skill registry.
        model_id: If set, rewrite ``model_id`` on each pinned arm.
        base_pinner: Creates general/data-only/existing configs.

    Returns:
        Mapping of general, data-only and existing-skill modes to pinned refs.
    """
    return base_pinner(registry)


def capture_live_config(
    registry: Registry,
    emulator_config: dict[str, Any],
    *,
    pinner: Callable[[Registry], dict[str, str]] = pin,
) -> dict[str, Any]:
    """Pin general/data-only/existing configs and return a secret-free manifest.

    Args:
        registry: Writable registry inside admin-ui.
        emulator_config: ``/control/config`` payload from the emulator.
        pinner: Returns general/data-only/existing refs; defaults to ``pin``.

    Returns:
        Manifest with refs, config bodies, prompt versions, skill hash and
        emulator condition. Secrets are never included.

    Raises:
        ValueError: If a pinned config is malformed or emulator fields are missing.
    """
    pinned = pinner(registry)
    refs = {
        BenchmarkMode.GENERAL_KNOWLEDGE.value: pinned["general_knowledge"],
        BenchmarkMode.SKILLS_DISABLED.value: pinned["skills_disabled"],
        BenchmarkMode.EXISTING_SKILLS.value: pinned["existing_skills"],
    }
    configs: dict[str, dict[str, Any]] = {}
    prompt_versions: dict[str, str] = {}
    for ref in refs.values():
        _version, raw = registry.load(ref)
        if not isinstance(raw, dict) or not isinstance(raw.get("system_prompt_ref"), str):
            raise ValueError(f"pinned agent config is malformed: {ref}")
        prompt_version, _prompt = registry.load(raw["system_prompt_ref"])
        configs[ref] = runtime_agent_config(raw).as_dict()
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

    Reads ``B2E_REGISTRY_DB``, ``HEIMDALL_URL``, ``B2E_MODEL`` and
    ``B2E_HARNESS`` from the process environment.
    """
    load_env()
    registry_path = require_env("B2E_REGISTRY_DB")
    emulator_url = require_env("HEIMDALL_URL")
    registry = Registry(registry_path)
    try:
        payload = capture_live_config(
            registry,
            fetch_json(f"{emulator_url.rstrip('/')}/control/config"),
            pinner=pin_live_configs,
        )
    finally:
        registry.close()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
