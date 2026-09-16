"""Input contracts for the skill benchmark."""

from .cases import BenchmarkCase, load_case, load_suite, write_suite_jsonl
from .modes import (
    BenchmarkMode, CommonConditions, ModeConfig, ModeConfigs, build_modes,
    write_mode_config,
)
from .catalog_snapshots import compose_catalog, snapshot_catalog, verify_snapshot
from .contracts import (
    PROMPT_RENDERER_VERSION, RESPONSE_PROTOCOL_VERSION,
    render_agent_query, response_contract_hash, response_schema,
    validate_response_contract,
)
from .preflight import PreflightResult, preflight_case, preflight_suite
from .execution import (
    ActivatedMode, AgentRequest, AgentTurn, PinnedConfigActivator,
    StandSessionExecutor, trace_observations,
)
from .results import ResultWriter, RunResult, summarize_results
from .runner import BenchmarkRunner
from .scoring import NormalizedAnswer, calculate_metrics, normalize_answer

__all__ = [
    "BenchmarkCase", "BenchmarkMode", "CommonConditions", "ModeConfig",
    "ModeConfigs",
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
