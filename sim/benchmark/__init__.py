"""Public package surface for the skill benchmark.

Submodules are imported on attribute access so ``python -m sim.benchmark.live_config``
inside the admin-ui image does not pull jsonschema. That package is on the host
and in deploy/requirements.txt, but older images may not have it yet.

Attributes:
    __all__: Names re-exported from submodules on first access.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BenchmarkCase", "BenchmarkMode", "CommonConditions", "ModeConfig",
    "ModeConfigs", "ModeStrategy",
    "ActivatedMode", "AgentRequest", "AgentTurn", "BenchmarkRunner",
    "NormalizedAnswer", "PinnedConfigActivator", "PreflightResult",
    "PROMPT_RENDERER_VERSION", "RESPONSE_PROTOCOL_VERSION",
    "ResultWriter", "RunResult", "StandSessionExecutor", "build_modes",
    "calculate_metrics", "compose_catalog", "load_case", "load_suite",
    "normalize_answer", "preflight_case", "preflight_suite",
    "render_agent_query", "response_contract_hash", "response_schema",
    "snapshot_catalog", "summarize_results", "trace_observations",
    "validate_response_contract", "verify_snapshot", "write_mode_config",
    "write_suite_jsonl",
]

_EXPORTS = {
    "ActivatedMode": ".execution",
    "AgentRequest": ".execution",
    "AgentTurn": ".execution",
    "BenchmarkCase": ".cases",
    "BenchmarkMode": ".modes",
    "BenchmarkRunner": ".runner",
    "CommonConditions": ".modes",
    "ModeConfig": ".modes",
    "ModeConfigs": ".modes",
    "ModeStrategy": ".modes",
    "NormalizedAnswer": ".scoring",
    "PinnedConfigActivator": ".execution",
    "PreflightResult": ".preflight",
    "PROMPT_RENDERER_VERSION": ".contracts",
    "RESPONSE_PROTOCOL_VERSION": ".contracts",
    "ResultWriter": ".results",
    "RunResult": ".results",
    "StandSessionExecutor": ".execution",
    "build_modes": ".modes",
    "calculate_metrics": ".scoring",
    "compose_catalog": ".catalog_snapshots",
    "load_case": ".cases",
    "load_suite": ".cases",
    "normalize_answer": ".scoring",
    "preflight_case": ".preflight",
    "preflight_suite": ".preflight",
    "render_agent_query": ".contracts",
    "response_contract_hash": ".contracts",
    "response_schema": ".contracts",
    "snapshot_catalog": ".catalog_snapshots",
    "summarize_results": ".results",
    "trace_observations": ".execution",
    "validate_response_contract": ".contracts",
    "verify_snapshot": ".catalog_snapshots",
    "write_mode_config": ".modes",
    "write_suite_jsonl": ".cases",
}


def __getattr__(name: str) -> Any:
    """Load a public export on first access.

    Args:
        name: Attribute listed in ``__all__``.

    Returns:
        The object exported by the owning submodule.

    Raises:
        AttributeError: If ``name`` is not a public export.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
