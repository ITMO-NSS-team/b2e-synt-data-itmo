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

**The shape depends on the harness, and the difference is not cosmetic.** Read
this before writing any query against the span store.

#### `harness=messages_api` — the in-process loop (`sim/agent/loop.py`)

Anthropic is called from this process, so every layer is visible:

```
AGENT  b2e.turn                                ← fingerprint lives here
├── CHAIN  iteration.1
│   ├── LLM    llm.messages.create             ← tokens, model, params
│   └── TOOL   tool.mcp_query
│       └── CHAIN  heimdall.mcp_query          ← status, latency, payloads
├── CHAIN  iteration.2
│   ├── LLM    llm.messages.create
│   └── TOOL   tool.run_skill
│       └── CHAIN  sandbox execute             ← skill hash, exit, limits
└── LLM    llm.messages.create                 ← final answer
```

#### `harness=claude_code` — the default (`sim/agent/claude_code.py`)

The model call happens **inside the Claude Code subprocess**. Nothing here is on
that call path — but the CLI writes a session transcript, and every assistant
entry in it carries `message.usage` *and the content blocks the model produced*.
Grouped by `message.id`, those are the API calls, one `LLM` span each. The
stream names the same `message.id` while the turn runs, which is what gives the
iteration level its boundaries:

```
AGENT  b2e.turn                                ← fingerprint + turn-level totals
├── CHAIN  iteration.1                         ← one model call and the tools it asked for
│   ├── LLM    llm.messages.create             ← tokens, reasoning, completion, measured ttft
│   └── TOOL   tool.mcp__heimdall__mcp_query   ← real start/end, input, output
│       └── CHAIN  heimdall.mcp_query          ← status, path, rows, bridge-measured
├── CHAIN  iteration.2
│   ├── LLM    llm.messages.create
│   └── TOOL   tool.Bash
│       └── CHAIN  sandbox.execute             ← skill hash, exit, limits
└── CHAIN  iteration.3
    └── LLM    llm.messages.create             ← stop_reason=end_turn
```

Verified on a live 14-iteration turn (2026-08-09, trace `d1f1df08…`): 11
`iteration.N` spans, 11 `LLM` spans one per iteration, 13 `TOOL` spans under the
iteration that asked for each, and 10 `heimdall.*` under their tool call.

**This is the same shape as `messages_api`.** Until 2026-08-09 this document
said the iteration level was unobservable outside the CLI. It is observable: the
CLI announces the message.

What is available at each level, and what is not:

| Want | `messages_api` | `claude_code` |
|---|---|---|
| per-model-call tokens | `LLM` span | `LLM` span, from the CLI's session transcript |
| per-call model, stop reason | `LLM` span | `LLM` span |
| per-call latency | measured | **derived** — see below |
| per-call time to first token | not emitted | **measured**, `b2e.llm.ttft_ms` |
| per-call reasoning | not emitted | `LLM` span, from the transcript |
| completion text per call | `LLM` span | `LLM` span, from the transcript |
| the prompt as sent | `LLM` span | **partial** — conversation yes, CLI system prompt and tool schemas no |
| tool latency | `TOOL` span | `TOOL` span, timed live off the event stream |
| tool input / output | `TOOL` span | `TOOL` span |
| HTTP status, path, row counts | Heimdall `CHAIN` | Heimdall `CHAIN`, from the bridge's own journal |
| skill digest, sandbox limits | sandbox `CHAIN` | sandbox `CHAIN`, from the runner's own output |
| iteration boundaries | `CHAIN iteration.N` | `CHAIN iteration.N`, from the stream |

Where each timing comes from, because they are not the same kind of number:

- **`TOOL`** — opened when the CLI announces `tool_use`, closed on the matching
  `tool_result` (`ToolSpanRecorder`). Measured live, in this process.
- **`heimdall.*`** — start and duration both measured *inside the bridge*,
  around the request, and carried out through its journal. This process only
  transports them.
- **`sandbox.execute`** — spans the whole Bash call, with the sandbox's own
  `b2e.sandbox.wall_ms` as an attribute. The window is the tool call's, because
  when within that call the code ran is not something anyone measured, and a
  fabricated timestamp beside a real one is worse than none.
- **`CHAIN iteration.N`** — measured live off the stream. The boundary between
  one iteration and the next is **when the previous iteration's last tool result
  arrived**, not when the next message was first seen: the gap between the two
  is the model thinking, and that belongs to the call it produced. Cutting at
  first sight instead would leave every `LLM` span starting fractionally before
  its own parent.
- **`LLM`** — **the window is derived, and says so; the first token is
  measured.** The transcript carries no duration of any kind (checked: no
  `ttft`, `duration`, `latency`, `elapsed`, `_ms`; and `diagnostics` is null).
  The window is the gap between the previous recorded row and the message's
  **last** row — two real stamps, but nothing timed the request. Every such span
  carries `b2e.llm.timing="derived"`. Do not compare it with a `messages_api`
  `LLM` duration without saying which is which.

  Separately, `b2e.llm.ttft_ms` **is** a measurement, taken by the CLI and
  reported on the `message_start` row that `--include-partial-messages`
  produces, keyed by the same `message.id`. It carries
  `b2e.llm.ttft_source="measured"` so the two numbers on one span are never
  confused for each other. Range on the live turn above: 824–4 387 ms.

Until 2026-08-08 the `TOOL` spans were replayed after the subprocess exited and
every one had a duration of zero, and the two `CHAIN` layers did not exist at
all. **A trace recorded before that date has no tool timing — not imprecise
timing, none.** Background and remaining work: `docs/subprocess-tracing-plan.md`.

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

On `harness=claude_code` the root additionally carries the turn's totals. These
are the CLI's own closing figures, independent of the per-call `LLM` spans
reconstructed from the transcript — which is what makes them a cross-check
rather than a duplicate:

| Attribute | Meaning |
|---|---|
| `llm.token_count.prompt` | everything the model read, **cached prefix included** |
| `llm.token_count.completion` | tokens generated |
| `llm.token_count.total` | the two above |
| `llm.token_count.prompt_details.cache_read` | of the prompt, served from cache |
| `llm.token_count.prompt_details.cache_write` | of the prompt, written to cache |
| `b2e.turn.uncached_prompt_tokens` | the raw `usage.input_tokens` |
| `b2e.turn.tool_time_ms` | wall time inside tool calls |
| `b2e.turn.duration_ms` | turn duration as the CLI measured it |
| `b2e.turn.iterations` / `.tool_calls` / `.heimdall_calls` / `.skill_runs` / `.cost_usd` | turn counters |
| `b2e.turn.iteration_spans` | how many `iteration.N` spans were opened; below `iterations` when the CLI counts a turn the stream did not announce |
| `b2e.turn.api_duration_ms` / `.ttft_ms` / `.ttft_stream_ms` / `.time_to_request_ms` | the CLI's own measurements, recorded as reported |

`api_duration_ms` is **not** a wall-clock slice of the turn and must not be
subtracted from one — on the live turn above it was 55 152 ms against a
`duration_ms` of 90 189 ms, and on a single-call probe it exceeded the turn
duration outright (5 732 against 3 898). It is the CLI's figure for time inside
model calls, whatever that includes; nothing here reinterprets it.

The standard `llm.token_count.*` keys are used rather than bespoke `b2e.*` ones
because they are the names every other tool already understands.

**They do not populate Phoenix's `llm_token_count_prompt` / `_completion`
columns, and there is no `span_costs` row.** Verified on a live turn
(2026-08-08, trace `dfabe77d…`): the attributes are all present and correct,
those columns are NULL, and `cumulative_llm_token_count_prompt` is 0. Phoenix
extracts them into columns for `LLM`-kind spans only, and this is an `AGENT`
span. Read them from `attributes`:

```sql
select attributes#>>'{llm,token_count,prompt}',
       attributes#>>'{llm,token_count,prompt_details,cache_read}'
from spans where span_kind = 'AGENT';
```

Emitting a synthetic child `LLM` span to satisfy the column would mean inventing
a model call that never happened, in a schema whose whole argument is that a
plausible number is worse than a missing one. The columns stay empty.

**`prompt` counts the cached prefix on purpose.** `usage.input_tokens` from the
CLI is only the part *not* served from cache — a real turn here reports
`input_tokens=30` beside `cache_read_input_tokens=24807`. Reporting the 30 as
the prompt size, which is what this stack did until 2026-08-08, understates it
by three orders of magnitude. The split is kept beside it because a cache read
and a fresh prompt token do not cost the same.

Model time is `b2e.turn.duration_ms - b2e.turn.tool_time_ms`. It is derived
rather than stored: storing it would imply this process measured it, and it
did not.

### `LLM`
`llm.model_name`, `llm.provider`, `llm.system`, `llm.invocation_parameters`,
`llm.input_messages`, `llm.output_messages`, `llm.token_count.prompt`,
`llm.token_count.completion`, `llm.token_count.total`, and where present
`llm.token_count.prompt_details.cache_read` / `.cache_write`.

Plus one local addition, `b2e.remaining_token_budget`: Haiku 4.5 receives a
live remaining-budget signal after each tool call. Whether the agent *uses* that
signal is a switchable strategy, so the trace records the value and
`b2e.budget_strategy` records whether it was consulted.

On `harness=claude_code` these spans are reconstructed from the CLI's session
transcript: model, provider, all five `llm.token_count.*` keys,
`llm.finish_reason`, plus `b2e.llm.message_id` (the Anthropic `msg_…` id) and
`b2e.llm.timing="derived"`.

Since 2026-08-09 they also carry what the model said. `llm.output_messages` is
written as **indexed flat attributes**, which Phoenix re-nests on ingest — a
JSON string is stored as a string and stays opaque, so the shape matters:

| Key | Value |
|---|---|
| `llm.output_messages.0.message.role` | `assistant` |
| `…contents.N.message_content.type` | `reasoning` or `text` |
| `…contents.N.message_content.text` | the reasoning, or the visible answer |
| `…contents.N.message_content.signature` | Anthropic's thinking signature |
| `…contents.N.message_content.data` | a `redacted_thinking` payload, when there is one |
| `…tool_calls.N.tool_call.{id,function.name,function.arguments}` | the calls this response asked for |
| `output.value` | the visible text **only** — reasoning is deliberately excluded, so a scorer scanning outputs for fabricated identifiers cannot find them in a passage the user never saw |

`reasoning` is the convention's own word for it; Anthropic's `thinking` is
translated at the parser and appears nowhere downstream.

`llm.input_messages` is present too and is **a reconstruction, not the
request** — it always travels with the flags that say so:

| Key | Meaning |
|---|---|
| `b2e.llm.prompt_reconstruction` | `conversation_only` |
| `b2e.llm.prompt_missing` | `cli_system_prompt,tool_schemas` |
| `llm.system` + `b2e.llm.system_partial` | the suffix this harness appended; Claude Code's own base prompt sits in front of it and is not exposed |
| `b2e.llm.input_messages_elided` | messages dropped from a long prefix, when any were |
| `b2e.trace.llm_content` | `transcript`, or `disabled` when capture is switched off |

The size of the hole is not small and is worth stating: on a real turn the first
call reported `input_tokens=10, cache_read=6526, cache_creation=5690` — about
12 200 prompt tokens for a 61-character question, of which the harness can
account for perhaps a thousand. **Do not read `llm.input_messages` on this
harness as the prompt that was billed.** Tool results inside a prefix are capped
and point at the `TOOL` span that has them in full.

Still absent: `llm.invocation_parameters` and `b2e.remaining_token_budget`.
Nothing is invented to fill those; do not write a query that assumes them on the
default harness.

Unlike the `AGENT` root, these **do** populate Phoenix's own `llm_token_count_*`
columns and produce `span_costs` rows — verified on a live turn (2026-08-08,
trace `34b09ef9…`: 11 `LLM` spans, 11 cost rows). The per-call prompt tokens sum
exactly to the turn total on the root, which is the check that the transcript is
being read once and not twice.

### `TOOL`
`tool.name`, `tool.description`, `tool.parameters`, and on the parent's message
`message.tool_calls` with `tool_call.function.name` /
`tool_call.function.arguments`.

On `harness=claude_code`, `tool.name` is the harness-side name the CLI reports
(`mcp__heimdall__mcp_query`, `Bash`, `ToolSearch`), the span is parented to the
`iteration.N` that asked for it — to the root only when the stream announced a
call without a message id — and one further attribute may appear:
`b2e.tool.unfinished=true`,
set when the turn ended before the tool returned. Such a span is closed and kept
rather than dropped — a hanging call is the one most worth seeing, and a missing
span is indistinguishable from a call that never happened.

### `CHAIN` (Heimdall HTTP)

Emitted on both harnesses, by different routes. On `messages_api` the call is
made here and the span wraps it. On `claude_code` the call is made by the MCP
bridge in its own process, which journals what it did to `HR_TRACE_LOG`; the
harness reads that back and materialises the span with the bridge's own start
time and duration.

Each record is filed under the `TOOL` span whose window contains it, matched by
tool name and claimed at most once — so paging a mart, which is the same tool
several times over, keeps one HTTP span per call rather than collapsing them.
A record that matches no tool call is still kept, parented to the turn and
marked `b2e.trace.correlation="unmatched"`: an unattributed call must still be
counted, and must not be mistaken for an attributed one.

This is what makes the tool-call-to-HTTP ratio readable from the trace. For the
2026-08-02..05 corpus, recorded before any of this existed, it was 1:1 —
established by cross-referencing the emulator's access log (Aug 4: 51 / 51;
Aug 5: 17 / 17), not by reading a trace.

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

On `claude_code` the span is named `sandbox.execute` and is emitted from the
runner's own stdout JSON, which is the Bash tool result. It appears only when
the Bash call was a skill run: `echo` and `pwd` are permitted regardless of the
allowlist, and manufacturing a sandbox span for them would invent executions
that never happened. `b2e.sandbox.exit_status` carries the runner's refusal kind
(`not-executable`, `timeout`, `bad-args`) or `ok`.

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
