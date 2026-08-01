# Observability: what was verified, and how

The spec requires verifying the OpenInference span attribute names and the
Phoenix client API **against the installed package versions**, rather than
trusting remembered field names. This document is that record.

Method: introspect the installed packages directly
(`/tmp/.../scratchpad/probe_oi.py`), not documentation and not memory.

---

## 1. Installed versions

Read from `importlib.metadata` in the project venv on 2026-08-01:

| Package | Version |
|---|---|
| `arize-phoenix-otel` | **0.16.1** |
| `openinference-semantic-conventions` | **0.1.31** |
| `openinference-instrumentation` | **0.1.56** |
| `openinference-instrumentation-anthropic` | **1.1.0** |
| `opentelemetry-sdk` | **1.44.0** |
| `opentelemetry-exporter-otlp-proto-http` | **1.44.0** |
| `anthropic` | **0.120.2** |

---

## 2. Span kinds actually present

`OpenInferenceSpanKindValues` members, enumerated from the installed enum:

```
TOOL  CHAIN  LLM  RETRIEVER  EMBEDDING  AGENT  RERANKER
UNKNOWN  GUARDRAIL  EVALUATOR  PROMPT
```

Note `GUARDRAIL`, `EVALUATOR` and `PROMPT` exist in 0.1.31 — worth knowing before
inventing local equivalents.

---

## 3. Attribute names confirmed

Selected `SpanAttributes` members, verified as `NAME = value` pairs:

| Constant | Wire key |
|---|---|
| `OPENINFERENCE_SPAN_KIND` | `openinference.span.kind` |
| `SESSION_ID` | `session.id` |
| `USER_ID` | `user.id` |
| `METADATA` | `metadata` |
| `INPUT_VALUE` / `OUTPUT_VALUE` | `input.value` / `output.value` |
| `INPUT_MIME_TYPE` / `OUTPUT_MIME_TYPE` | `input.mime_type` / `output.mime_type` |
| `LLM_MODEL_NAME` | `llm.model_name` |
| `LLM_PROVIDER` / `LLM_SYSTEM` | `llm.provider` / `llm.system` |
| `LLM_INVOCATION_PARAMETERS` | `llm.invocation_parameters` |
| `LLM_INPUT_MESSAGES` / `LLM_OUTPUT_MESSAGES` | `llm.input_messages` / `llm.output_messages` |
| `LLM_TOKEN_COUNT_PROMPT` | `llm.token_count.prompt` |
| `LLM_TOKEN_COUNT_COMPLETION` | `llm.token_count.completion` |
| `LLM_TOKEN_COUNT_TOTAL` | `llm.token_count.total` |
| `LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ` | `llm.token_count.prompt_details.cache_read` |
| `LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE` | `llm.token_count.prompt_details.cache_write` |
| `TOOL_NAME` / `TOOL_DESCRIPTION` / `TOOL_PARAMETERS` | `tool.name` / `tool.description` / `tool.parameters` |
| `LLM_TOOLS` | `llm.tools` |

`MessageAttributes`:

| Constant | Wire key |
|---|---|
| `MESSAGE_ROLE` | `message.role` |
| `MESSAGE_CONTENT` | `message.content` |
| `MESSAGE_CONTENTS` | `message.contents` |
| `MESSAGE_TOOL_CALLS` | `message.tool_calls` |
| `MESSAGE_TOOL_CALL_ID` | `message.tool_call_id` |

`ToolCallAttributes` — note these do **not** follow the `tool.*` prefix:

| Constant | Wire key |
|---|---|
| `TOOL_CALL_ID` | `tool_call.id` |
| `TOOL_CALL_FUNCTION_NAME` | `tool_call.function.name` |
| `TOOL_CALL_FUNCTION_ARGUMENTS_JSON` | `tool_call.function.arguments` |

That last one is a real trap: the constant is named `..._JSON` but the wire key
has no `_json` suffix. Writing `tool_call.function.arguments_json` from memory
would produce spans Phoenix does not recognise, and the failure is silent.

### Cost attributes exist natively

0.1.31 ships `llm.cost.prompt`, `llm.cost.completion`, `llm.cost.total`, and a
`llm.cost.prompt_details.*` family including `cache_read` / `cache_write`. The
cost guard therefore populates standard keys instead of a local invention.

---

## 4. Phoenix registration API

`phoenix.otel.register` signature as installed:

```python
register(*, endpoint=None, project_name=None, batch=False,
         set_global_tracer_provider=True, headers=None,
         protocol=None,            # 'http/protobuf' | 'grpc'
         verbose=True, auto_instrument=False, api_key=None,
         **kwargs) -> TracerProvider
```

Exports available from `phoenix.otel`: `register`, `TracerProvider`,
`BatchSpanProcessor`, `SimpleSpanProcessor`, `HTTPSpanExporter`,
`GRPCSpanExporter`, `Resource`, `SpanAttributes`,
`OpenInferenceSpanKindValues`, `OpenInferenceMimeTypeValues`,
`suppress_tracing`, and the context managers `using_session`, `using_user`,
`using_attributes`, `using_metadata`, `using_tags`, `using_prompt_template`.

Consequences taken:

- `batch=True` in the services. The default is `False` (a `SimpleSpanProcessor`,
  one HTTP round trip per span), which on 2 vCPU would put export latency inside
  the measurement.
- `using_session(session_id)` and `using_user(user_id)` are used rather than
  setting `session.id` / `user.id` by hand, so nested spans inherit them.
- `auto_instrument=False`. Instrumentation is explicit, because an implicit one
  that silently stops matching a library version would leave a trace that looks
  complete and is not.

---

## 5. Feedback as annotations

Feedback is stored through Phoenix's annotation API rather than a local table, as
the spec requires. `research-api` is the only writer.

`arize-phoenix-otel` is the tracing half only; annotation writes go to the
Phoenix server's REST API (`POST /v1/span_annotations`). Depending on the
`arize-phoenix` server package for this would pull the whole server into the
research-api image for one call, which does not fit the memory budget — so the
call is made over HTTP against the running Phoenix.

---

## 6. Deliberate non-verification

Two things are **not** verified here and are called out rather than assumed:

1. **Phoenix server version compatibility.** The server runs from the
   `arize-phoenix` container image; the annotation endpoint path is asserted by
   `make smoke` against the running instance rather than trusted from docs. If
   the smoke check fails, that is the contract having moved.
2. **Cost figures.** `llm.cost.*` is populated from a local price table in
   config, not from the API. Anthropic pricing is not exposed per response, so
   any cost number here is a *projection* from configured rates — the cost guard
   documents its own assumption.
