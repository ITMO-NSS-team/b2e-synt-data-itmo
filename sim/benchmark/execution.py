"""One isolated agent turn and trace-derived observations.

No gold fields enter this module's request contract.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .modes import BenchmarkMode, ModeConfig, catalog_hash

_PINNED_REF = re.compile(r"^[^@\s]+@[1-9][0-9]*$")


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Isolated turn request. Gold never appears here.

    Attributes:
        query: Rendered user message, including the public response schema.
        employee_id: Runtime actor for Heimdall scope.
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

    This is sufficient for skills_disabled and heimdall_skills, which share
    the same mounted standard catalog. A future non-mock generated mode needs a
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
        if mode.name == BenchmarkMode.GENERATED_SKILL and not mode.is_mock:
            raise ValueError("generated_skill requires a catalog-mounting ModeActivator")
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
        return AgentTurn(
            answer=turn.answer,
            stats={
                **turn.stats,
                "heimdall_calls": turn.heimdall_calls,
                "tool_calls": turn.tool_calls,
                "total_tokens": turn.total_tokens,
                "latency_ms": turn.latency_ms,
            },
            trace=turn.trace,
            error=(
                turn.error
                or (
                    f"trace unavailable for session {turn.session_id}"
                    if turn.session_id and turn.trace is None else None
                )
            ),
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
    root_attrs: dict[str, Any] = {}
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
        status = _attr(attrs, "b2e.http.status")
        if isinstance(status, (int, float)):
            http_statuses.append(int(status))
        error_code = _attr(attrs, "b2e.heimdall.error_code")
        if error_code:
            error_codes.append(str(error_code))
        if name == "get_skill" and isinstance(arguments, dict) and arguments.get("name"):
            loaded.add(str(arguments["name"]))
        if name == "find_skills":
            found.update(_skill_names(output))
        if name == "mcp_query":
            target = bridge_mcp if _attr(attrs, "b2e.heimdall.endpoint") else tool_mcp
            target.append(span)
            rows = _attr(attrs, "b2e.heimdall.rows")
            if status is not None and isinstance(rows, (int, float)):
                mcp_query_rows.append(int(rows))
        if _failed_span(span, attrs) and _is_tool_span(span, attrs):
            if _attr(attrs, "b2e.heimdall.endpoint"):
                failed_bridge.append(span)
            else:
                failed_tool.append((span, attrs))

    stats = turn.stats
    return {
        "found_skills": sorted(found),
        "loaded_skills": sorted(loaded),
        "tool_calls": _int(stats.get("tool_calls"), _attr(root_attrs, "b2e.turn.tool_calls")),
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
        "total_tokens": _int(
            stats.get("total_tokens"), _attr(root_attrs, "llm.token_count.total")
        ),
        "latency_ms": _number(stats.get("latency_ms")),
        "agent_duration_ms": _number(_attr(root_attrs, "b2e.turn.duration_ms")),
        "tool_time_ms": _number(_attr(root_attrs, "b2e.turn.tool_time_ms")),
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
