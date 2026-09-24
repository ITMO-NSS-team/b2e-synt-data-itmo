"""OpenTelemetry / OpenInference instrumentation.

Attribute names here are the ones read out of the installed packages, not
remembered ones — see ``docs/observability.md`` for the verification record and
``docs/span-schema.md`` for the schema this emits. Two of them are genuine traps:
``ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON`` has the wire key
``tool_call.function.arguments`` with no ``_json`` suffix, and span kinds are
enum *values*, not free strings. Getting either wrong produces spans Phoenix
quietly fails to interpret.

Enforcement
-----------
:func:`start_run` refuses to open a root span without a complete fingerprint.
That is the spec's requirement and it is here rather than at the call site
because a check you have to remember to call is a check that eventually is not
called.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from sim.fingerprint import RunFingerprint, require_complete

try:
    from openinference.semconv.trace import (
        MessageAttributes,
        MessageContentAttributes,
        OpenInferenceSpanKindValues,
        SpanAttributes,
        ToolCallAttributes,
    )
except ImportError:                                       # pragma: no cover
    raise RuntimeError(
        "openinference-semantic-conventions is required; "
        "pip install openinference-semantic-conventions"
    )

#: Single attribute cap. A 642-column `SELECT *` would otherwise dominate the
#: span store. Truncated values are flagged so nobody mistakes a clipped payload
#: for a short one.
MAX_ATTR_BYTES = 128 * 1024
TRUNCATED_FLAG = "b2e.truncated"

SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND

_TRACER_NAME = "b2e.sim"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


def configure(
    *,
    endpoint: str,
    project_name: str,
    protocol: str = "http/protobuf",
    batch: bool = True,
    secondary_endpoint: str | None = None,
    service_name: str = "b2e-agent",
) -> Any:
    """Register a tracer provider for Phoenix and optionally another OTLP sink.

    ``batch=True`` deliberately overrides the library default of ``False``. The
    default installs a SimpleSpanProcessor, which performs one HTTP round trip
    per span; on 2 vCPU that would put export latency inside the thing being
    measured.

    ``auto_instrument`` is left off. Implicit instrumentation that silently
    stops matching a library version leaves a trace that looks complete and is
    not — and this environment exists to compare traces.
    """
    from phoenix.otel import register
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME

    provider = register(
        endpoint=endpoint,
        project_name=project_name,
        protocol=protocol,
        batch=batch,
        auto_instrument=False,
        set_global_tracer_provider=True,
        verbose=False,
        resource=Resource.create({SERVICE_NAME: service_name}),
    )
    if secondary_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )

        target = _trace_endpoint(secondary_endpoint)
        if target != endpoint.rstrip("/"):
            processor = BatchSpanProcessor if batch else SimpleSpanProcessor
            provider.add_span_processor(
                processor(OTLPSpanExporter(endpoint=target)),
                replace_default_processor=False,
            )
    return provider


def _trace_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"


# ------------------------------------------------------------------ helpers


def _encode(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def set_attr(span: Span, key: str, value: Any) -> None:
    """Set one attribute, truncating oversized payloads visibly."""
    if value is None:
        return
    if isinstance(value, (bool, int, float)):
        span.set_attribute(key, value)
        return
    text = _encode(value)
    raw = text.encode("utf-8")
    if len(raw) > MAX_ATTR_BYTES:
        text = raw[:MAX_ATTR_BYTES].decode("utf-8", "ignore")
        span.set_attribute(TRUNCATED_FLAG, True)
    span.set_attribute(key, text)


def set_io(span: Span, *, input_value: Any = None, output_value: Any = None) -> None:
    """Record input/output with the right mime type for each."""
    if input_value is not None:
        set_attr(span, SpanAttributes.INPUT_VALUE, input_value)
        set_attr(span, SpanAttributes.INPUT_MIME_TYPE,
                 "text/plain" if isinstance(input_value, str) else "application/json")
    if output_value is not None:
        set_attr(span, SpanAttributes.OUTPUT_VALUE, output_value)
        set_attr(span, SpanAttributes.OUTPUT_MIME_TYPE,
                 "text/plain" if isinstance(output_value, str) else "application/json")


# ----------------------------------------------------------------- messages

#: Content types the convention names. Read out of the installed
#: ``MessageContentAttributes`` docstring, which lists `"text"`, `"image"`,
#: `"audio"`, `"reasoning"`, `"tool_use"`. A provider's own word — Anthropic
#: says `thinking` — would leave every other OpenInference reader unable to find
#: it, so the mapping happens at the edge that knows the provider.
REASONING = "reasoning"
TEXT = "text"


def set_messages(span: Span, collection: str,
                 messages: list[dict[str, Any]]) -> None:
    """Write messages as **indexed flat attributes**, not one JSON blob.

    ``collection`` is ``SpanAttributes.LLM_INPUT_MESSAGES`` or
    ``LLM_OUTPUT_MESSAGES``. Each entry may carry ``role``, ``content`` (a
    plain string), ``contents`` (a list of ``{type, text, signature, data}``),
    ``tool_calls`` (a list of ``{id, name, arguments}``) and ``tool_call_id``.

    Why flat and not a blob
    -----------------------
    Both forms are accepted by the exporter and only one is understood. Checked
    against the deployed Phoenix (19.13.0) on 2026-08-09 by emitting the same
    span twice: the flat keys are re-nested on ingest into
    ``llm.output_messages[0].message.contents[…].message_content.type``, which
    is what the UI renders and what any other OpenInference reader expects,
    while a JSON string is stored as a string and stays opaque.

    ``record_llm_result`` still writes the blob form on the ``messages_api``
    harness. That is a known defect kept out of this change's blast radius — it
    is the same wrong shape those spans have always had, and fixing it changes
    a corpus this one does not touch.
    """
    for i, message in enumerate(messages):
        at = f"{collection}.{i}."
        role = message.get("role")
        if role:
            span.set_attribute(at + MessageAttributes.MESSAGE_ROLE, str(role))
        if message.get("content") is not None:
            set_attr(span, at + MessageAttributes.MESSAGE_CONTENT,
                     message["content"])
        if message.get("tool_call_id"):
            span.set_attribute(at + MessageAttributes.MESSAGE_TOOL_CALL_ID,
                               str(message["tool_call_id"]))
        for j, item in enumerate(message.get("contents") or []):
            content_at = f"{at}{MessageAttributes.MESSAGE_CONTENTS}.{j}."
            span.set_attribute(
                content_at + MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                str(item.get("type") or TEXT))
            set_attr(span, content_at + MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                     item.get("text"))
            set_attr(span,
                     content_at + MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE,
                     item.get("signature"))
            # `redacted_thinking` arrives as opaque bytes with no text at all.
            # The convention has a field for exactly this and names Anthropic in
            # its docstring, so a redacted block stays a reasoning block that
            # happens to be unreadable rather than becoming a missing one.
            set_attr(span,
                     content_at + MessageContentAttributes.MESSAGE_CONTENT_DATA,
                     item.get("data"))
        for j, call in enumerate(message.get("tool_calls") or []):
            call_at = f"{at}{MessageAttributes.MESSAGE_TOOL_CALLS}.{j}."
            if call.get("id"):
                span.set_attribute(call_at + ToolCallAttributes.TOOL_CALL_ID,
                                   str(call["id"]))
            span.set_attribute(call_at + ToolCallAttributes.TOOL_CALL_FUNCTION_NAME,
                               str(call.get("name") or ""))
            set_attr(span,
                     call_at + ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                     call.get("arguments"))


# -------------------------------------------------------------------- spans


@contextmanager
def start_run(
    name: str,
    *,
    fingerprint: RunFingerprint | None,
    session_id: str,
    employee_id: str,
    metadata: dict[str, Any] | None = None,
    question: str | None = None,
    memory_ref: str | None = None,
    memory_digest: str | None = None,
    memory_epoch: int | None = None,
    memory_tokens: int | None = None,
    run_arm: str | None = None,
    run_epoch: int | None = None,
    run_replication: int | None = None,
) -> Iterator[Span]:
    """Root AGENT span for one turn.

    Refuses to open without a complete fingerprint: a run recorded without one
    cannot be compared with any other run, so producing it is worse than
    failing — it looks like data.

    The ``memory_*``/``run_*`` keywords are RQ4's addition, following the
    ``b2e.code_execution`` precedent (``sim/agent/claude_code.py``): none of
    them are a tenth fingerprint field, because ``memory_ref`` already travels
    inside ``agent_config_version`` and would double-count the condition if it
    were. They exist so a researcher can filter a trace store on the arm
    directly, without resolving a config version first. All are optional and
    unset by default — ``run_arm``/``run_epoch``/``run_replication`` name
    concepts (which of several fleet instances, which reflection epoch) that
    belong to the Plan C driver that assembles them; this call only accepts
    the values, it does not invent them.
    """
    fingerprint = require_complete(fingerprint)

    with get_tracer().start_as_current_span(name) as span:
        span.set_attribute(SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        span.set_attribute(SpanAttributes.SESSION_ID, session_id)
        span.set_attribute(SpanAttributes.USER_ID, str(employee_id))
        benchmark_run_id = (metadata or {}).get("run_id")
        span.set_attribute(
            "b2e.run.id",
            str(benchmark_run_id or f"{span.get_span_context().trace_id:032x}"),
        )
        span.set_attribute(
            "b2e.run.kind",
            "benchmark_case" if benchmark_run_id else "business_task",
        )
        for key, value in fingerprint.as_span_attributes().items():
            span.set_attribute(key, value)
        span.set_attribute("b2e.run.condition_id", fingerprint.condition_id)
        if metadata:
            set_attr(span, SpanAttributes.METADATA, metadata)
            for key in (
                "eval_id", "case_id", "mode", "run_id", "comparison_group_id",
            ):
                set_attr(span, f"b2e.benchmark.{key}", metadata.get(key))
        if question is not None:
            set_io(span, input_value=question)
        if memory_ref is not None:
            span.set_attribute("b2e.memory.ref", memory_ref)
        if memory_digest is not None:
            span.set_attribute("b2e.memory.digest", memory_digest)
        if memory_epoch is not None:
            span.set_attribute("b2e.memory.epoch", int(memory_epoch))
        if memory_tokens is not None:
            span.set_attribute("b2e.memory.tokens", int(memory_tokens))
        if run_arm is not None:
            span.set_attribute("b2e.run.arm", run_arm)
        if run_epoch is not None:
            span.set_attribute("b2e.run.epoch", int(run_epoch))
        if run_replication is not None:
            span.set_attribute("b2e.run.replication", int(run_replication))
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


@contextmanager
def start_chain(name: str, **attrs: Any) -> Iterator[Span]:
    with get_tracer().start_as_current_span(name) as span:
        span.set_attribute(SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        for key, value in attrs.items():
            set_attr(span, key, value)
        yield span


@contextmanager
def start_llm(
    name: str,
    *,
    model: str,
    invocation_parameters: dict[str, Any],
    messages: list[dict[str, Any]],
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[Span]:
    with get_tracer().start_as_current_span(name) as span:
        span.set_attribute(SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
        span.set_attribute(SpanAttributes.LLM_PROVIDER, "anthropic")
        set_attr(span, SpanAttributes.LLM_SYSTEM, system)
        set_attr(span, SpanAttributes.LLM_INVOCATION_PARAMETERS, invocation_parameters)
        set_attr(span, SpanAttributes.LLM_INPUT_MESSAGES, messages)
        if tools:
            set_attr(span, SpanAttributes.LLM_TOOLS, tools)
        set_io(span, input_value=messages)
        yield span


def record_llm_result(
    span: Span,
    *,
    output_messages: list[dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    cache_read: int | None = None,
    cache_write: int | None = None,
    stop_reason: str | None = None,
    remaining_budget: int | None = None,
    budget_strategy: str | None = None,
    cost_usd: float | None = None,
) -> None:
    set_attr(span, SpanAttributes.LLM_OUTPUT_MESSAGES, output_messages)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(prompt_tokens))
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(completion_tokens))
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                       int(prompt_tokens) + int(completion_tokens))
    if cache_read is not None:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, int(cache_read))
    if cache_write is not None:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, int(cache_write))
    if cost_usd is not None:
        span.set_attribute(SpanAttributes.LLM_COST_TOTAL, float(cost_usd))
    set_attr(span, SpanAttributes.LLM_FINISH_REASON, stop_reason)
    # Haiku 4.5 reports remaining budget after each tool call. Whether the agent
    # consults it is a switchable strategy, so record both the signal and the
    # decision to use it.
    if remaining_budget is not None:
        span.set_attribute("b2e.remaining_token_budget", int(remaining_budget))
    set_attr(span, "b2e.budget_strategy", budget_strategy)
    set_io(span, output_value=output_messages)


def _describe_tool(span: Span, name: str, description: str,
                   parameters: dict[str, Any] | None,
                   tool_call_id: str | None) -> None:
    span.set_attribute(SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
    span.set_attribute(SpanAttributes.TOOL_NAME, name)
    set_attr(span, SpanAttributes.TOOL_DESCRIPTION, description)
    set_attr(span, SpanAttributes.TOOL_PARAMETERS, parameters)
    if tool_call_id:
        span.set_attribute(ToolCallAttributes.TOOL_CALL_ID, tool_call_id)
    set_io(span, input_value=parameters)


@contextmanager
def start_tool(
    name: str,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    tool_call_id: str | None = None,
) -> Iterator[Span]:
    with get_tracer().start_as_current_span(f"tool.{name}") as span:
        _describe_tool(span, name, description, parameters, tool_call_id)
        yield span


def open_tool_span(
    name: str,
    *,
    parent: Span,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    tool_call_id: str | None = None,
) -> Span:
    """A TOOL span opened now and closed later, explicitly parented.

    The context-manager form cannot be used where a tool starts and ends in
    different callbacks — which is every tool run by an out-of-process harness,
    where the only evidence of the call is two events arriving on a stream.

    ``parent`` is passed as a context rather than taken from the ambient one on
    purpose: the events are read on the thread draining the CLI's stdout, and
    OTel's current-span context is thread-local. Relying on the ambient context
    there produces spans in a brand-new trace, which looks like a working
    instrumentation right up until you try to read a turn.
    """
    ctx = trace.set_span_in_context(parent)
    span = get_tracer().start_span(f"tool.{name}", context=ctx)
    _describe_tool(span, name, description, parameters, tool_call_id)
    return span


def close_tool_span(span: Span, *, output: Any = None,
                    unfinished: bool = False) -> None:
    """End a span opened by :func:`open_tool_span`.

    ``unfinished`` marks a tool whose result never arrived — the turn timed out
    or the CLI died mid-call. The span is still closed rather than dropped: a
    hanging call is the one you most need to see, and a missing span is
    indistinguishable from a call that never happened.
    """
    if output is not None:
        set_io(span, output_value=output)
    if unfinished:
        span.set_attribute("b2e.tool.unfinished", True)
        span.set_status(Status(StatusCode.ERROR, "no tool_result before the turn ended"))
    span.end()


@contextmanager
def start_heimdall_call(
    *,
    method: str,
    path: str,
    endpoint: str,
    request_body: Any = None,
) -> Iterator[Span]:
    """One HTTP call to Heimdall, nested under the TOOL that caused it.

    Separate from the tool span because one tool call can become several HTTP
    calls, and "API calls per answer" — an RQ2 metric — is exactly that ratio.
    """
    with get_tracer().start_as_current_span(f"heimdall.{endpoint}") as span:
        span.set_attribute(SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute("b2e.http.method", method)
        span.set_attribute("b2e.http.path", path)
        span.set_attribute("b2e.heimdall.endpoint", endpoint)
        set_io(span, input_value=request_body)
        yield span


def record_heimdall_result(
    span: Span,
    *,
    status: int,
    body: Any,
    rows: int = 0,
    columns_requested: int = 0,
    injected_latency_ms: float | None = None,
) -> None:
    span.set_attribute("b2e.http.status", int(status))
    span.set_attribute("b2e.heimdall.rows", int(rows))
    span.set_attribute("b2e.heimdall.columns_requested", int(columns_requested))
    if injected_latency_ms is not None:
        # Kept separate from real elapsed time: a p95 that blends simulated and
        # actual latency is a number nobody can interpret.
        span.set_attribute("b2e.latency.injected_ms", float(injected_latency_ms))
    if isinstance(body, dict) and "code" in body:
        set_attr(span, "b2e.heimdall.error_code", body.get("code"))
    if status >= 400:
        span.set_status(Status(StatusCode.ERROR, f"HTTP {status}"))
    set_io(span, output_value=body)


def record_skill_execution(
    span: Span,
    *,
    skill_name: str,
    skill_hash: str,
    state: str,
    exit_status: str,
    wall_ms: float,
    peak_rss_kb: int | None = None,
    rejected_imports: list[str] | None = None,
) -> None:
    """``skill_hash`` is the digest actually executed, recorded after the runner
    re-verified it — so the trace proves which bytes ran, not which name was
    asked for."""
    span.set_attribute("b2e.skill.name", skill_name)
    span.set_attribute("b2e.skill.hash", skill_hash)
    span.set_attribute("b2e.skill.state", state)
    span.set_attribute("b2e.sandbox.exit_status", exit_status)
    span.set_attribute("b2e.sandbox.wall_ms", float(wall_ms))
    if peak_rss_kb is not None:
        span.set_attribute("b2e.sandbox.peak_rss_kb", int(peak_rss_kb))
    if rejected_imports:
        set_attr(span, "b2e.sandbox.rejected_imports", rejected_imports)


def current_traceparent() -> str | None:
    """The current span as a W3C ``traceparent``, for handing to a subprocess.

    Version 00, sampled flag always 01: this environment does not sample — a
    sampled trace is not a comparable one — so a subprocess is never told to
    drop what it records.
    """
    ctx = trace.get_current_span().get_span_context()
    if not ctx or not ctx.trace_id or not ctx.span_id:
        return None
    return f"00-{format(ctx.trace_id, '032x')}-{format(ctx.span_id, '016x')}-01"


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx and ctx.trace_id else None


def current_span_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.span_id, "016x") if ctx and ctx.span_id else None
