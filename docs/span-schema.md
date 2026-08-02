# Span schema

**Gate G2.** What every trace must contain for two runs to be comparable.

Attribute names below are **not from memory**. They were read out of the
installed packages by introspection; see `docs/observability.md` for the
verification transcript and versions.

---

## 1. Span kinds

`openinference.span.kind` is set from `OpenInferenceSpanKindValues`. The
installed enum (`openinference-semantic-conventions` 0.1.31) offers: `AGENT`,
`CHAIN`, `LLM`, `TOOL`, `RETRIEVER`, `EMBEDDING`, `RERANKER`, `GUARDRAIL`,
`EVALUATOR`, `PROMPT`, `UNKNOWN`.

This environment uses five:

| Kind | Emitted for | One per |
|---|---|---|
| `AGENT` | the whole turn — **root span** | user message |
| `CHAIN` | a reasoning/tool-loop iteration | loop pass |
| `LLM` | one model call | request to Anthropic |
| `TOOL` | one tool invocation by the agent | tool call |
| `CHAIN` | one Heimdall HTTP call, nested under its `TOOL` | HTTP request |

Heimdall calls are `CHAIN` rather than `TOOL` on purpose: the *tool* is what the
agent chose to do, the HTTP call is how it was carried out. One tool call can
become several HTTP calls, and RQ2 is largely about that ratio. Collapsing them
would make "API calls per answer" unmeasurable.

### Tree shape per turn

```
AGENT  turn                                    ← fingerprint lives here
├── CHAIN  iteration 1
│   ├── LLM    messages.create                 ← tokens, model, params
│   └── TOOL   mcp_query
│       └── CHAIN  POST /api/v1/mcp/query/     ← status, latency, payloads
├── CHAIN  iteration 2
│   ├── LLM    messages.create
│   └── TOOL   run_skill
│       └── CHAIN  sandbox execute             ← skill hash, exit, limits
└── LLM    final answer
```

---

## 2. The run fingerprint

**Every root span carries all nine fields. A run with any field unset does not
start.** Enforced in `sim/fingerprint.py`; `RunFingerprint.create` raises
`IncompleteFingerprint` before a token is spent, and a blank string counts as
unset because that is how a missing environment variable actually arrives.

| Attribute | Type | Source |
|---|---|---|
| `b2e.run.agent_config_version` | str | registry ref, e.g. `agent_config@3` |
| `b2e.run.prompt_registry_version` | str | registry ref, e.g. `system_prompt@7` |
| `b2e.run.skill_registry_hash` | str | SHA-256 over active skill hashes |
| `b2e.run.model_id` | str | e.g. `claude-haiku-4-5-20251001` |
| `b2e.run.temperature` | float | sampling temperature |
| `b2e.run.data_snapshot_hash` | str | snapshot manifest id |
| `b2e.run.traps_enabled` | bool | RQ1 independent variable |
| `b2e.run.latency_profile` | str | `instant` / `realistic` / `degraded` |
| `b2e.run.hr_employee_ids` | str | sorted ids holding the HR role, or `none` |

Flat scalar keys, not a nested object: OTel attribute values are scalars, and
Phoenix filters on flat keys.

`hr_employee_ids` is a sorted comma-joined string rather than a list, for that
same reason, and `none` rather than `""` because an empty string is what this
module treats as unset — while "nobody holds HR" is the default condition and
has to be expressible. It is read from the emulator's `/control/healthz`, not
asserted locally: the service that enforces the grant is the one that gets to
report it.

`traps_enabled` and `data_snapshot_hash` are both required and are not
redundant. Traps exist at two layers — channel quirks toggle per request, but the
data traps in `b2e/traps.py` are baked into the corpus at build time. A traps-off
run is served from a *different snapshot*. Only the pair identifies the condition.

`condition_id` (SHA-256 over all eight, first 16 hex chars) is derived, not
stored: two runs are comparable exactly when their `condition_id` matches.

---

## 3. Required attributes by kind

### All spans
`openinference.span.kind`, `input.value`, `output.value`, and the matching
`input.mime_type` / `output.mime_type`.

### `AGENT` (root)
- all eight `b2e.run.*` fields
- `session.id` — the B2E session id
- `user.id` — the employee id
- `metadata` — JSON: `{employee_role, question_id, experiment_id, basket_id}`

`session.id` and `user.id` are the real OpenInference keys (`SpanAttributes.SESSION_ID`
= `"session.id"`, `SpanAttributes.USER_ID` = `"user.id"`), which is what makes
Phoenix group traces into sessions natively instead of needing a bespoke table.

### `LLM`
`llm.model_name`, `llm.provider`, `llm.system`, `llm.invocation_parameters`,
`llm.input_messages`, `llm.output_messages`, `llm.token_count.prompt`,
`llm.token_count.completion`, `llm.token_count.total`, and where present
`llm.token_count.prompt_details.cache_read` / `.cache_write`.

Plus one local addition, `b2e.remaining_token_budget`: Haiku 4.5 receives a
live remaining-budget signal after each tool call. Whether the agent *uses* that
signal is a switchable strategy, so the trace records the value and
`b2e.budget_strategy` records whether it was consulted.

### `TOOL`
`tool.name`, `tool.description`, `tool.parameters`, and on the parent's message
`message.tool_calls` with `tool_call.function.name` /
`tool_call.function.arguments`.

### `CHAIN` (Heimdall HTTP)
`b2e.http.method`, `b2e.http.path`, `b2e.http.status`, `b2e.heimdall.endpoint`
(the logical operation, e.g. `mcp_query`), `b2e.heimdall.error_code` when the
envelope carries one, `b2e.heimdall.rows`, `b2e.heimdall.columns_requested`,
`b2e.latency.injected_ms`.

`b2e.latency.injected_ms` is separated from real elapsed time deliberately.
Reporting a p95 that silently blends simulated and actual latency would be a
number nobody could interpret.

### `CHAIN` (sandbox execute)
`b2e.skill.hash`, `b2e.skill.name`, `b2e.skill.state`, `b2e.sandbox.exit_status`,
`b2e.sandbox.wall_ms`, `b2e.sandbox.peak_rss_kb`, `b2e.sandbox.rejected_imports`.

`b2e.skill.hash` is the SHA-256 actually executed, recorded after the runner
re-verifies the digest — so a trace proves *which bytes ran*, not which name was
requested.

---

## 4. Feedback

Feedback is written as **Phoenix annotations**, not a bespoke table, so it lives
beside the trace it describes and survives independent of this codebase.

| Level | Target | Fields |
|---|---|---|
| response | span id | `label` (`like`/`dislike`), `score` (1/0), `explanation` (free text) |
| session | session id | same, applied at session scope |

Annotation name is `user_feedback`. `research-api` `POST /feedback` writes it;
nothing else does.

The distinction between "liked" and "correct" is the point — `docs/research-agenda.md`
notes that likes correlate with confident tone rather than accuracy. Oracle
scoring is recorded separately as an annotation named `oracle`, so the divergence
between them is directly queryable.

---

## 5. Sampling and payload size

No sampling. Every span is exported; this is an experiment, not production, and a
sampled trace is not a comparable one.

Payloads are recorded in full, as the spec requires, with one guard: a single
attribute is truncated at 128 KiB and marked `b2e.truncated=true`. A 642-column
`SELECT *` response would otherwise dominate the span store.
