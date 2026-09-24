"""One isolated agent turn and trace-derived observations.

No gold fields enter this module's request contract.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .modes import ModeConfig, catalog_hash

_PINNED_REF = re.compile(r"^[^@\s]+@[1-9][0-9]*$")
# Claude Code reports every tool rejected by the active allow-list as
# ``denied:<tool name>``.  Built-ins are named ``Bash``/``Write`` while MCP
# tools use qualified lowercase names such as
# ``mcp__heimdall__find_skills``.  Both are successful enforcement of the
# benchmark arm, not transport failures.
_HARNESS_DENIAL = re.compile(r"^denied:[A-Za-z0-9_.:/-]+$")

OPERATIONAL_METRICS = (
    "iterations", "tool_calls", "heimdall_calls", "mcp_query_calls",
    "failed_tool_calls", "permission_denials", "http_error_count",
    "mcp_query_rows_total", "heimdall_response_bytes", "prompt_tokens",
    "uncached_prompt_tokens", "cache_read_tokens", "cache_creation_tokens",
    "completion_tokens", "total_tokens", "cache_hit_ratio", "cost_usd",
    "latency_ms", "agent_duration_ms", "api_duration_ms", "ttft_ms",
    "ttft_stream_ms", "time_to_request_ms", "tool_time_ms", "tool_time_ratio",
)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Isolated turn request. Gold never appears here.

    Attributes:
        query: Rendered user message, including the public response schema.
        employee_id: Runtime actor for information-service scope.
        config_ref: Pinned agent config, e.g. ``agent_config_benchmark_skills_on@2``.
        metadata: Non-secret run labels copied into the stand session.
    """
    query: str
    employee_id: str
    config_ref: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """Raw outcome of one stand session.

    Attributes:
        answer: Final model text.
        stats: Duration, token and tool counters from the stand client.
        trace: Phoenix-style span tree, if retrieved.
        error: Transport or missing-trace error; ``None`` on a finished turn.
            Successful harness denials such as ``denied:Bash`` stay on the
            stand payload but are not treated as a failed turn.
        session_id: Stand session id.
        trace_id: Root trace id, when present.
        fingerprint: Live experiment fingerprint reported by the agent.
    """
    answer: str
    stats: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] | None = None
    error: str | None = None
    session_id: str | None = None
    trace_id: str | None = None
    fingerprint: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ActivatedMode:
    """Pinned config actually used for a mode after activation checks.

    Attributes:
        config_ref: Registry ref the agent will load.
        catalog_hash: Skill catalog hash for this arm, or ``None``.
    """
    config_ref: str
    catalog_hash: str | None


def fatal_turn_error(error: str | None, *, answer: str) -> str | None:
    """Keep transport failures; drop successful harness denials when an answer exists.

    The agent reports ``denied:Bash`` after Claude Code refuses a forbidden
    tool. That is the intended boundary, not a broken session: the model still
    finished the turn. Mixed or non-denial errors stay fatal.

    Args:
        error: Stand ``errors`` payload, already stringified.
        answer: Final model text for this turn.

    Returns:
        The original error when the cell should be ``unscored``; ``None``
        when only harness denials remain and ``answer`` is non-empty.
    """
    if not error:
        return None
    items = _error_items(error)
    if (
        items
        and all(_HARNESS_DENIAL.fullmatch(item) for item in items)
        and answer.strip()
    ):
        return None
    return error


def _error_items(error: str) -> list[str]:
    try:
        value = ast.literal_eval(error)
    except (SyntaxError, ValueError):
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    return []


class SessionExecutor(Protocol):
    """An implementation must open a fresh session for every call."""

    def execute(self, request: AgentRequest) -> AgentTurn:
        """Run one isolated turn.

        Args:
            request: Query, actor and pinned config for this sample.

        Returns:
            Answer, stats and optional trace. Transport failures may set
            ``error`` instead of raising.
        """
        ...


class ModeActivator(Protocol):
    """Swap live agent config (and catalog, if needed) around a turn."""

    def activate(self, mode: ModeConfig) -> ActivatedMode:
        """Make ``mode`` the live agent configuration.

        Args:
            mode: Arm to enable.

        Returns:
            Pinned ref and catalog hash actually in force.
        """
        ...

    def deactivate(self, mode: ModeConfig) -> None:
        """Undo activation. No-op is allowed when the stand is already isolated.

        Args:
            mode: Arm that was activated.
        """
        ...


class PinnedConfigActivator:
    """Use already-created pinned agent configs.

    This is sufficient for general_knowledge and existing_skills. The former
    has no catalog access; the latter uses the mounted standard catalog. A future
    non-mock generated mode needs a
    deployment-specific activator that mounts its combined catalog first.
    """

    def __init__(
        self,
        config_refs: Mapping[str, str],
        config_reader: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        """Bind mode names to already-pinned registry refs.

        Args:
            config_refs: Mapping from mode name to pinned ``name@N`` ref.
            config_reader: Loads the live agent body for a ref; required on
                ``activate`` so tool subset and model can be verified.
        """
        self._refs = dict(config_refs)
        self._config_reader = config_reader

    def activate(self, mode: ModeConfig) -> ActivatedMode:
        """Verify the pinned live config matches ``mode``.

        Args:
            mode: Arm to enable.

        Returns:
            The pinned ref and catalog hash.

        Raises:
            ValueError: If the ref is missing, tools/model drift, or a
                non-mock generated mode needs a catalog-mounting activator.
        """
        ref = self._refs.get(mode.name)
        if not ref or not _PINNED_REF.fullmatch(ref):
            raise ValueError(f"{mode.name}: pinned config_ref is required")
        if self._config_reader is None:
            raise ValueError(f"{mode.name}: config_reader is required to verify actual agent tools")
        config = self._config_reader(ref)
        if tuple(config.get("tool_subset") or ()) != mode.tool_subset:
            raise ValueError(f"{mode.name}: live agent tool_subset differs from mode")
        if (
            config.get("model_id") != mode.common.model_id
            or config.get("temperature") != mode.common.temperature
            or config.get("code_execution") != mode.common.code_execution
            or config.get("conversation_mode") != "stateless"
        ):
            raise ValueError(f"{mode.name}: live agent behavior differs from mode")
        if mode.strategy.requires_catalog_activator and not mode.is_mock:
            raise ValueError(
                f"{mode.name} requires a catalog-mounting ModeActivator"
            )
        if mode.catalog_path is not None:
            actual = catalog_hash(mode.catalog_path)
            if actual != mode.catalog_hash:
                raise ValueError(f"{mode.name}: catalog changed before activation")
        return ActivatedMode(ref, mode.catalog_hash)

    def deactivate(self, mode: ModeConfig) -> None:
        """No-op: pinned configs do not need teardown.

        Args:
            mode: Arm that was activated; unused.
        """
        del mode


class StandSessionExecutor:
    """Adapter over the stand client already used by sim.skill_eval.

    Import is lazy: defining and unit-testing the benchmark does not require
    optional HTTP dependencies or a running stand.
    """

    def __init__(self, **stand_options: Any) -> None:
        """Create the HTTP adapter lazily.

        Args:
            **stand_options: Keyword arguments for ``sim.skill_eval.stand.StandClient``.
        """
        from sim.skill_eval.stand import StandClient
        self._stand = StandClient(**stand_options)

    def execute(self, request: AgentRequest) -> AgentTurn:
        """Open a stand session, run one question, map the eval turn back.

        Args:
            request: Isolated query with no gold fields.

        Returns:
            Agent turn including trace when the stand can fetch it.
        """
        from sim.skill_eval.types import EvalCase, SessionSpec

        case = EvalCase(
            case_id=request.metadata["case_id"],
            category="",
            question=request.query,
            runtime_actor_employee_id=request.employee_id,
            expected_skill=None,
            expected_skill_kind=None,
            gold={},
            snapshot_id=None,
            business_task={},
            raw={},
        )
        turn = self._stand.run(
            case,
            SessionSpec(
                employee_id=request.employee_id,
                config_ref=request.config_ref,
                metadata=dict(request.metadata),
            ),
        )
        heimdall_calls = (
            0
            if request.metadata.get("heimdall_access") == "disabled"
            else turn.heimdall_calls
        )
        return AgentTurn(
            answer=turn.answer,
            stats={
                **turn.stats,
                "heimdall_calls": heimdall_calls,
                "tool_calls": turn.tool_calls,
                "total_tokens": turn.total_tokens,
                "latency_ms": turn.latency_ms,
            },
            trace=turn.trace,
            error=turn.error,
            session_id=turn.session_id,
            trace_id=turn.trace_id,
            fingerprint=turn.fingerprint,
        )


def trace_observations(turn: AgentTurn) -> dict[str, Any]:
    """Extract routing, call, error and timing facts from a full trace.

    Args:
        turn: Completed or failed session, including optional span tree.

    Returns:
        Observation dict consumed by ``calculate_metrics``: skill names,
        call counts, HTTP statuses, error codes and timings.
    """
    spans = list(_span_dicts(turn.trace))
    found: set[str] = set()
    loaded: set[str] = set()
    bridge_mcp = []
    tool_mcp = []
    failed_bridge = []
    failed_tool: list[tuple[dict[str, Any], dict[str, Any]]] = []
    http_statuses: list[int] = []
    error_codes: list[str] = []
    mcp_query_rows: list[int] = []
    heimdall_response_bytes: list[int] = []
    root_attrs: dict[str, Any] = {}
    denial_count = 0
    for span in spans:
        attrs = span.get("attributes") or {}
        name = _tool_name(span, attrs)
        if not root_attrs and any(str(key).startswith("b2e.turn.") for key in attrs):
            root_attrs = attrs
        arguments = _decode(
            _attr(attrs, "input.value")
            or _attr(attrs, "tool.parameters")
            or _attr(attrs, "tool_parameters")
        )
        output = _decode(_attr(attrs, "output.value"))
        denied = _permission_denied(output)
        status = _attr(attrs, "b2e.http.status")
        if isinstance(status, (int, float)):
            http_statuses.append(int(status))
        error_code = _attr(attrs, "b2e.heimdall.error_code")
        if error_code:
            error_codes.append(str(error_code))
        if (
            name == "get_skill"
            and isinstance(arguments, dict)
            and arguments.get("name")
            and not denied
        ):
            loaded.add(str(arguments["name"]))
        if denied and _is_tool_span(span, attrs):
            denial_count += 1
        if name == "find_skills":
            found.update(_skill_names(output))
        if name == "mcp_query":
            target = bridge_mcp if _attr(attrs, "b2e.heimdall.endpoint") else tool_mcp
            target.append(span)
            rows = _attr(attrs, "b2e.heimdall.rows")
            if status is not None and isinstance(rows, (int, float)):
                mcp_query_rows.append(int(rows))
        response_bytes = _attr(attrs, "b2e.heimdall.response_bytes")
        if isinstance(response_bytes, (int, float)):
            heimdall_response_bytes.append(int(response_bytes))
        if _failed_span(span, attrs) and _is_tool_span(span, attrs):
            if _attr(attrs, "b2e.heimdall.endpoint"):
                failed_bridge.append(span)
            else:
                failed_tool.append((span, attrs))

    stats = turn.stats
    prompt_tokens = _int(
        stats.get("prompt_tokens"), _attr(root_attrs, "llm.token_count.prompt")
    )
    cache_read_tokens = _int(
        _attr(root_attrs, "llm.token_count.prompt_details.cache_read")
    )
    agent_duration_ms = _number(_attr(root_attrs, "b2e.turn.duration_ms"))
    tool_time_ms = _number(_attr(root_attrs, "b2e.turn.tool_time_ms"))
    recorded_denials = _attr(root_attrs, "b2e.permission_denials")
    return {
        "found_skills": sorted(found),
        "loaded_skills": sorted(loaded),
        "tool_calls": _int(
            stats.get("tool_calls"), _attr(root_attrs, "b2e.turn.tool_calls")
        ),
        "heimdall_calls": _int(
            stats.get("heimdall_calls"), _attr(root_attrs, "b2e.turn.heimdall_calls")
        ),
        "mcp_query_calls": len(bridge_mcp or tool_mcp),
        "failed_tool_calls": (
            len(failed_bridge)
            + sum(not _looks_like_heimdall_tool(span, attrs) for span, attrs in failed_tool)
            if failed_bridge else len(failed_tool)
        ),
        "http_statuses": http_statuses,
        "error_codes": error_codes,
        "mcp_query_rows": mcp_query_rows,
        "mcp_query_rows_total": sum(mcp_query_rows),
        "heimdall_response_bytes": sum(heimdall_response_bytes),
        "iterations": _int(
            stats.get("iterations"), _attr(root_attrs, "b2e.turn.iterations")
        ),
        "permission_denials": (
            denial_count if recorded_denials is None else _int(recorded_denials)
        ),
        "http_error_count": sum(status >= 400 for status in http_statuses),
        "prompt_tokens": prompt_tokens,
        "uncached_prompt_tokens": _int(
            _attr(root_attrs, "b2e.turn.uncached_prompt_tokens")
        ),
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": _int(
            _attr(root_attrs, "llm.token_count.prompt_details.cache_write")
        ),
        "completion_tokens": _int(
            stats.get("completion_tokens"),
            _attr(root_attrs, "llm.token_count.completion"),
        ),
        "total_tokens": _int(
            stats.get("total_tokens"), _attr(root_attrs, "llm.token_count.total")
        ),
        "cache_hit_ratio": (
            cache_read_tokens / prompt_tokens if prompt_tokens else None
        ),
        "cost_usd": _number(
            stats.get("cost_usd")
            if stats.get("cost_usd") is not None
            else _attr(root_attrs, "b2e.turn.cost_usd")
        ),
        "latency_ms": _number(stats.get("latency_ms")),
        "agent_duration_ms": agent_duration_ms,
        "api_duration_ms": _number(_attr(root_attrs, "b2e.turn.api_duration_ms")),
        "ttft_ms": _number(_attr(root_attrs, "b2e.turn.ttft_ms")),
        "ttft_stream_ms": _number(_attr(root_attrs, "b2e.turn.ttft_stream_ms")),
        "time_to_request_ms": _number(
            _attr(root_attrs, "b2e.turn.time_to_request_ms")
        ),
        "tool_time_ms": tool_time_ms,
        "tool_time_ratio": (
            tool_time_ms / agent_duration_ms
            if tool_time_ms is not None and agent_duration_ms else None
        ),
    }


def _span_dicts(value: Any):
    if isinstance(value, dict):
        if "name" in value or "attributes" in value:
            yield value
        for key, child in value.items():
            if key != "attributes":
                yield from _span_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _span_dicts(child)


def _tool_name(span: dict[str, Any], attrs: dict[str, Any]) -> str:
    raw = (
        _attr(attrs, "b2e.heimdall.endpoint")
        or _attr(attrs, "tool.name")
        or span.get("name")
        or ""
    )
    return str(raw).split("__")[-1].split(".")[-1]


def _is_tool_span(span: dict[str, Any], attrs: dict[str, Any]) -> bool:
    return bool(
        _attr(attrs, "b2e.heimdall.endpoint")
        or _attr(attrs, "tool.name")
        or str(span.get("name") or "").startswith(("heimdall.", "mcp__"))
    )


def _looks_like_heimdall_tool(span: dict[str, Any], attrs: dict[str, Any]) -> bool:
    return "heimdall" in str(
        _attr(attrs, "tool.name") or span.get("name") or ""
    ).lower()


def _permission_denied(output: Any) -> bool:
    """Claude Code don't-ask refusals are plain text on the tool span."""
    return isinstance(output, str) and "has been denied" in output.lower()


def _failed_span(span: dict[str, Any], attrs: dict[str, Any]) -> bool:
    status = _attr(attrs, "b2e.http.status")
    status_code = (span.get("status") or {}).get("status_code")
    return (
        (isinstance(status, (int, float)) and status >= 400)
        or bool(_attr(attrs, "b2e.heimdall.error_code"))
        or str(status_code or "").upper() == "ERROR"
    )


def _skill_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        results = value.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict) and item.get("name"):
                    names.add(str(item["name"]))
        for child in value.values():
            names.update(_skill_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_skill_names(child))
    return names


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _attr(attributes: dict[str, Any], name: str) -> Any:
    if name in attributes:
        return attributes[name]
    current: Any = attributes
    for part in name.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _int(*values: Any) -> int:
    for value in values:
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 0


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
