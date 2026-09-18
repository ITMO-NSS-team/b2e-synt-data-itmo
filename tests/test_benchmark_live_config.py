"""Live stand manifests are captured without starting Docker or an LLM."""
from __future__ import annotations

import os

import pytest

from sim.benchmark.env import load_env, require_env
from sim.benchmark.live_config import capture_live_config, pin_live_configs
from sim.benchmark.modes import DATA_TOOLS, SKILL_TOOLS
from sim.registry import Registry


def test_capture_pins_two_configs_and_records_actual_condition(tmp_path) -> None:
    registry = Registry(tmp_path / "registry.db")
    registry.commit(
        "system_prompt", "prompt", {"template": "test"}, actor="test",
    )
    base = {
        "model_id": "light-model", "temperature": 0.0,
        "system_prompt_ref": "system_prompt", "tool_subset": list(SKILL_TOOLS),
        "code_execution": "forbidden", "conversation_mode": "stateless",
    }

    def pinner(target: Registry) -> dict[str, str]:
        enabled = target.commit("on", "agent", base, actor="test")
        disabled_body = dict(base)
        disabled_body["tool_subset"] = list(DATA_TOOLS)
        disabled = target.commit("off", "agent", disabled_body, actor="test")
        return {"skills_on": enabled.ref, "skills_off": disabled.ref}

    payload = capture_live_config(registry, {
        "data_snapshot_hash": "snapshot@test",
        "traps_enabled": True,
        "latency_profile": "instant",
        "hr_employee_ids": ["9", "7"],
        "ignored_operator_field": "not persisted",
    }, pinner=pinner)
    disabled = payload["refs"]["skills_disabled"]
    enabled = payload["refs"]["existing_skills"]
    assert tuple(payload["configs"][disabled]["tool_subset"]) == DATA_TOOLS
    assert tuple(payload["configs"][enabled]["tool_subset"]) == SKILL_TOOLS
    assert payload["prompt_versions"][disabled].startswith("system_prompt@")
    assert payload["skill_registry_hash"].startswith("sha256:")
    assert payload["emulator"] == {
        "data_snapshot_hash": "snapshot@test",
        "hr_employee_ids": ["9", "7"],
        "latency_profile": "instant",
        "traps_enabled": True,
    }
    registry.close()


def test_model_override_is_pinned_without_changing_source_config(tmp_path) -> None:
    registry = Registry(tmp_path / "registry.db")
    source = {
        "model_id": "heavy-model", "system_prompt_ref": "system_prompt",
        "tool_subset": list(SKILL_TOOLS),
    }
    source_version = registry.commit("source", "agent", source, actor="test")

    def pinner(target: Registry) -> dict[str, str]:
        _version, raw = target.load(source_version.ref)
        on = target.commit("on", "agent", raw, actor="test")
        off_body = dict(raw)
        off_body["tool_subset"] = list(DATA_TOOLS)
        off = target.commit("off", "agent", off_body, actor="test")
        return {"skills_on": on.ref, "skills_off": off.ref}

    refs = pin_live_configs(
        registry, model_id="light-model", base_pinner=pinner,
    )
    assert registry.load(source_version.ref)[1]["model_id"] == "heavy-model"
    assert {registry.load(ref)[1]["model_id"] for ref in refs.values()} == {
        "light-model"
    }
    registry.close()


def test_load_env_reads_local_file_and_process_overrides(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'B2E_REGISTRY_DB="/from-file"\nHEIMDALL_URL=http://from-file:1\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("B2E_REGISTRY_DB", raising=False)
    monkeypatch.delenv("HEIMDALL_URL", raising=False)
    load_env(env_file)
    assert os.getenv("B2E_REGISTRY_DB") == "/from-file"
    assert os.getenv("HEIMDALL_URL") == "http://from-file:1"
    monkeypatch.setenv("HEIMDALL_URL", "http://from-process:2")
    load_env(env_file)
    assert os.getenv("HEIMDALL_URL") == "http://from-process:2"


def test_require_env_refuses_blank_instead_of_docker_defaults(monkeypatch) -> None:
    monkeypatch.delenv("B2E_REGISTRY_DB", raising=False)
    with pytest.raises(ValueError, match="B2E_REGISTRY_DB is empty"):
        require_env("B2E_REGISTRY_DB")
    monkeypatch.setenv("HEIMDALL_URL", "  ")
    with pytest.raises(ValueError, match="HEIMDALL_URL is empty"):
        require_env("HEIMDALL_URL")


