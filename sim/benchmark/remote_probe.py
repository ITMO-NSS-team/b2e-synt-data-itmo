"""SSH probe. Returns hashes/settings/checks, never skill contents.

The registry probe append-only pins ``general_knowledge`` and
``existing_skills`` from the live ``agent_config``. Catalog validation stays
read-only. The local driver appends shared validators and an entry point
before sending this source to Python's stdin.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import re
import sys

from heimdall.catalog.model import Catalog
from heimdall.engine.compile import compile_query
from heimdall.skills.registry import EXTENSIONS, Registry

_JSON_FENCE = re.compile(r"[\x60]{3}(json|jsonc)\s*\n(.*?)\n[\x60]{3}", re.IGNORECASE | re.DOTALL)


def probe(kind: str, options: dict) -> dict:
    import httpx

    load_env()
    if kind == "registry":
        from sim.registry import Registry as ConfigRegistry
        from sim.skills import SkillStore

        registry = ConfigRegistry(require_env("B2E_REGISTRY_DB"))
        try:
            from sim.skill_eval.pin import pin

            pinned = pin(registry, source_ref=options.get("config_ref") or "agent_config")
            keys = ("model_id", "temperature", "harness", "tool_subset",
                    "code_execution", "conversation_mode", "system_prompt_ref",
                    "skill_registry_ref")
            configs = {}
            prompt_versions = {}
            for ref in pinned.values():
                _version, config = registry.load(ref)
                prompt_version, _prompt = registry.load(config["system_prompt_ref"])
                configs[ref] = {key: config.get(key) for key in keys}
                prompt_versions[ref] = prompt_version.ref
            result = {
                "schema_version": "1.0",
                "phoenix_project": os.environ.get("PHOENIX_PROJECT", "b2e-itmo"),
                "refs": pinned,
                "configs": configs,
                "prompt_versions": prompt_versions,
                "skill_registry_hash": SkillStore(registry).registry_hash(),
            }
        finally:
            registry.close()
        base = require_env("HEIMDALL_URL")
        with httpx.Client(base_url=base, timeout=30, trust_env=False) as client:
            response = client.get("/control/config")
            response.raise_for_status()
            result["emulator"] = response.json()
            scopes = {}
            for employee in options["employee_ids"]:
                response = client.get("/control/scope", headers={"X-Employee-Id": str(employee)})
                response.raise_for_status()
                scope = response.json()
                scopes[str(employee)] = {key: scope[key] for key in ("employee_id", "role")}
        return {"live": result, "scopes": scopes}

    if kind == "catalog":
        skills = Path(require_env("HEIMDALL_SKILLS"))
        catalog = Path(require_env("HEIMDALL_CATALOG"))
        with httpx.Client(timeout=30, trust_env=False) as client:
            # Loopback of this container's own listener, not a Compose DNS default.
            response = client.get("http://127.0.0.1:8081/control/config")
            response.raise_for_status()
            condition = response.json()
        root = Path(condition["snapshot"])
        # These shared functions are appended by the local driver.
        before = catalog_hash(skills)
        registry = validate_skill_catalog(skills, Catalog.load(catalog))
        after = catalog_hash(skills)
        if before != after:
            raise ValueError("skill catalog changed during validation")
        if not registry.all_names():
            raise ValueError("remote skill catalog is empty")
        return {
            "emulator": condition,
            "snapshot_id": json.loads((root / "manifest.json").read_text())["snapshot_id"],
            "catalog_path": str(skills), "catalog_hash": after,
            "model_catalog_hash": "sha256:" + hashlib.sha256(catalog.read_bytes()).hexdigest(),
            "validated": True, "skill_count": len(registry.all_names()),
        }
    raise ValueError(f"unknown probe: {kind}")
