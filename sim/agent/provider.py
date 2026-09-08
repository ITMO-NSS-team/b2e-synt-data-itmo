"""LLM provider settings read only from the process environment."""
from __future__ import annotations

import math
import os


ZAI_PROVIDERS = frozenset({"zai", "z.ai", "zhipu", "glm"})

_ZAI_PASSTHROUGH = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "API_TIMEOUT_MS",
)


def llm_provider() -> str:
    return (os.environ.get("LLM_PROVIDER") or "").strip().lower()


def is_zai() -> bool:
    return llm_provider() in ZAI_PROVIDERS


def env_model_id() -> str:
    return (os.environ.get("B2E_MODEL") or "").strip()


def env_harness() -> str:
    return (os.environ.get("B2E_HARNESS") or "").strip()


def turn_timeout_seconds(fallback: int = 600) -> int:
    raw = (os.environ.get("B2E_TURN_TIMEOUT") or "").strip()
    timeout = int(raw) if raw else fallback
    api_ms = (os.environ.get("API_TIMEOUT_MS") or "").strip()
    if api_ms:
        timeout = max(timeout, math.ceil(int(api_ms) / 1000))
    return timeout


def zai_child_env() -> dict[str, str]:
    token = (
        (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
        or (os.environ.get("ZAI_API_KEY") or "").strip()
    )
    if not token:
        raise RuntimeError(
            "LLM_PROVIDER=zai requires ANTHROPIC_AUTH_TOKEN or ZAI_API_KEY")
    base_url = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError("LLM_PROVIDER=zai requires ANTHROPIC_BASE_URL")
    env = {
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_BASE_URL": base_url,
    }
    for key in _ZAI_PASSTHROUGH:
        if key == "ANTHROPIC_BASE_URL":
            continue
        value = (os.environ.get(key) or "").strip()
        if value:
            env[key] = value
    return env
