# Tracing the reasoning and the intermediate prompts

**Status:** stages 0, 1, 2, 3 and 4 landed 2026-08-09 and are **deployed**.
Stage 5 is not a tracing change and stays a research question. Successor to
`docs/subprocess-tracing-plan.md`, which closed every layer of the tree except
this one.

That document ends with a section titled "What stays unmeasurable, and saying
so", and its first sentence is:

> Even with all four stages, the trace never sees inside the model: no
> reasoning, no per-token timing, no view of why one query shape was chosen over
> another.

**The first clause is wrong, and has been wrong since the day
`--no-session-persistence` was dropped.** The reasoning is on disk, in a file
this code already opens, parses, and then deletes. This document is the
correction and the plan to act on it.

---

## 1. What was measured before anything was written

All figures below are from the live stack on 2026-08-09, not from a fixture.
Sixteen real B2E session transcripts under
`/app/var/claude-home/.claude/projects/*/*.jsonl` inside `b2e-sim-b2e-agent-1`,
CLI 2.1.220, every row `entrypoint: "sdk-cli"`, every assistant row
`model: claude-haiku-4-5-20251001`.

| Measured | Value |
|---|---|
| assistant rows | 751 |
| `thinking` content blocks | **291** |
| of those, with non-empty text | **291 — all of them** |
| transcripts containing reasoning | **16 / 16** |
| transcripts with assistant rows but no reasoning | **0** |
| reasoning bytes | 208 472 total; p50 588, p95 1 899, max 7 438 per block |
| `text` blocks | 152, 65 569 bytes |
| `tool_use` blocks | 308 |
| `tool_result` blocks | 308, 713 288 bytes; p50 252, p95 19 854, max 50 268 |
| `attachment` rows | 56, all `deferred_tools_delta`, ~399 bytes each |

A verbatim sample, from `1e4e0ec1`, on a question about headcount:

> «Пользователь спрашивает, сколько всего сотрудников в компании и сколько
> подразделений. Мне нужно получить эти данные через API Heimdall. Согласно
> инструкциям, мне нужно сначала загрузить инструменты Heimdall через
> ToolSearch. […] После этого я смогу запросить данные о количестве
> сотрудников»

That is the agent choosing a query shape, in its own words. It is the primary
subject matter of RQ2 — "агент вынужден добывать ответ формой запроса" — and
the bench has been discarding it on every turn.

### Why it looked unusable

Because on an interactive session it is. Across 409 transcripts on the
development host, 2 214 `thinking` blocks split cleanly on one field:

| `entrypoint` | non-empty thinking | empty thinking |
|---|---|---|
| `sdk-cli` (headless `claude -p`) | **all** | none, except one fixture |
| `cli` (interactive TUI) | none | **all 1 956** |

Interactive Claude Code persists the block with its `signature` and an empty
`thinking` string. Headless does not redact. Anyone who checked their own
`~/.claude` would conclude the field is always blank and stop. The B2E harness
runs headless, exclusively.

### Where it is thrown away

Three places, all in `sim/agent/claude_code.py`:

1. `parse_transcript` (`:846`) reads each assistant row, takes `message.usage`,
   and never touches `message.content`. `LlmCall` (`:798`) has no field that
   could hold it.
2. The same function drops every row after the first for a given `message.id`
   (`if not message_id or message_id in calls: continue`, `:901`) — and the
   later rows are precisely where `text` and `tool_use` live.
3. `_read_transcript` (`:689`) then `unlink()`s the file and `rmtree()`s its
   directory for any turn that is neither resumable nor an experiment.

And one more, upstream of all of them: `parse_stream` (`:1019`) and
`ToolSpanRecorder.observe` (`:1226`) both walk assistant events block by block
and keep only `tool_use`. The reasoning goes past them live, too.

---

## 2. The structure is better than the schema doc assumes

`docs/span-schema.md` says per-iteration `CHAIN` spans are impossible on this
harness and that `LLM` timing is derived from "the gap between the previous
recorded row and the response's own timestamp". The real file supports more than
that. One turn, verbatim:

```
22:38:29.211  assistant  msg_011CdeiwNGXR5ZfMT2naytcp  req_011CdeiwGk9…  ['thinking']  in=10 out=355 cr=6526  cw=5690
22:38:30.037  assistant  msg_011CdeiwNGXR5ZfMT2naytcp  req_011CdeiwGk9…  ['tool_use']
22:38:30.145  user       tool_result toolu_01C8hn…  422 B
22:38:33.073  assistant  msg_011CdeiwgabmF2Q45Vctafqo  req_011Cdeiwfh2…  ['thinking']  in=10 out=259 cr=12216 cw=2355
22:38:33.489  assistant  msg_011CdeiwgabmF2Q45Vctafqo  req_011Cdeiwfh2…  ['text']
22:38:33.490  assistant  msg_011CdeiwgabmF2Q45Vctafqo  req_011Cdeiwfh2…  ['tool_use']
22:38:33.599  user       tool_result toolu_01D5zZ…  19 854 B
```

Four facts follow, none of which the current parser uses:

- **One row per content block, each with its own timestamp.** The CLI writes a
  block when that block is finalised during streaming. So the first row of a
  message is approximately when the model's first output block completed, and
  the last row is approximately when the message ended. The current span ends at
  the *first* row, which for a reasoning-first model means the `LLM` span closes
  before the answer was written.
- **`requestId` (`req_…`) is stable across the rows of one `message.id`** and is
  a second correlation key, independent of the Anthropic message id.
- **The full conversation is present in order**: user question, assistant
  thinking/text/tool_use, tool results with their `tool_use_id`, interleaved
  correctly even when one message issues two parallel tool calls.
- **`toolUseResult` also appears as a top-level row key** beside the message
  block, so tool output is stored twice and neither copy is truncated at these
  sizes.

Reasoning arrives *before* the tool call it justifies, in the same message. That
is the causal link the trace has been missing: not "the agent called `mcp_query`
with these columns", but "the agent decided to call `mcp_query` with these
columns *because*…".

---

## 3. Measurement, reconstruction, and the part that stays dark

The repo's rule is that a plausible number is worse than a missing one. Applying
it honestly here means splitting the request into three tiers.

### Tier 1 — real, verbatim, no interpretation

The model's **output**: reasoning text, its signature, assistant text, tool_use
names and arguments, stop reason, per-call token counts. All of it is what the
API returned, written down by the CLI. Recording it is transcription.

The **conversation** as it stood at each call: the question, prior assistant
messages, tool results. Also verbatim.

### Tier 2 — reconstruction, and must be labelled as one

The **prompt** for call N. The harness knows its own `system_prompt` and the
`harness_note` it appended, and the transcript supplies the conversation. But
that is not what was sent.

Measured on the first call of a real turn: `input_tokens=10`,
`cache_read=6 526`, `cache_creation=5 690` — about **12 200 prompt tokens for a
61-character question**. The harness's own system prompt and note are on the
order of a thousand tokens. So roughly **ten thousand tokens, over 80% of that
first request, are Claude Code's base system prompt and its tool schemas**, and
the transcript does not contain them. The `attachment` rows carry only
`deferred_tools_delta`, ~399 bytes.

A span that presents the reconstruction as `llm.input_messages` without saying
so would mislead exactly the analysis this bench exists to support: anyone
studying token cost or prompt sensitivity would be reading a prompt four fifths
of which is absent.

### Tier 3 — genuinely unmeasurable from outside

Per-token timing. Time to first token, unless the live stream is opened up
(stage 4). The exact bytes on the wire. Whatever the model considered and did
not write down.

---

## 4. Is statelessness the obstacle?

**No. It is not, and it never was.**

The question is worth answering precisely because the code's own documentation
still gets it wrong. `sim/agent/config.py` says of `stateless`:

> For ``claude_code`` that is ``--no-session-persistence``

That is stale. The flag was removed, and the module docstring in
`sim/agent/claude_code.py:47` records why and what was checked:

> What enforces that is the absence of `--resume`, not the absence of a file on
> disk. […] Checked on the live stack before the flag was dropped: two turns in
> the same working directory, persistence on, no `--resume`, and the second had
> no memory of the first.

So the position today is: **turns are isolated by not passing `--resume`, and
the transcript is written anyway.** Every one of the 291 reasoning blocks
measured above was produced by a stack that is already stateless in the sense
the experiments require.

Consequently:

| Change | Touches a fingerprint field? | Changes what the model sees? |
|---|---|---|
| Reading content out of the transcript instead of only tokens | no | **no** |
| Not deleting the transcript afterwards | no | **no** |
| `--include-partial-messages` | no | no — an output-stream flag |
| `--system-prompt` instead of `--append-system-prompt` | **yes, effectively** | **yes** |
| `CLAUDE_CODE_ENABLE_TELEMETRY` | no | no, but emits metrics only (`4a`, settled) |
| hooks | conflicts with `--setting-sources ""` | possibly |

Stages 1–3 below are all in the first two rows. They are pure transcription of
data the CLI already wrote, in a process that is not on the model's call path.
Nothing about them requires, or benefits from, giving up statelessness.

**The separate question — should experiment arms be stateless at all — is
unaffected and the current answer stands.** A batch arm must stay stateless
because turns sharing context are not independent samples and per-turn token
counts stop being comparable once turn *n* pays to re-read turns 1..*n*−1. That
argument is about experimental design, not about observability, and this plan
does not touch it. `conversation_mode=resume` gets richer traces for free
alongside, because a resumed session's transcript holds every turn and the
`since` cutoff in `parse_transcript` already separates them.

---

## 5. Where this lands in Phoenix

The installed `openinference-semantic-conventions` is **0.1.31** — the same
version `docs/span-schema.md` was verified against — and it already has
first-class reasoning support. Read out of
`/usr/local/lib/python3.12/site-packages/openinference/semconv/trace/__init__.py`
in the agent container, not from memory:

```
MessageContentAttributes.MESSAGE_CONTENT_TYPE      = "message_content.type"
    """The type of the content, such as "text", "image", "audio",
       "reasoning", or "tool_use"."""
MessageContentAttributes.MESSAGE_CONTENT_TEXT      = "message_content.text"
    """The text content of the message, if the type is "text" or "reasoning"."""
MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE = "message_content.signature"
    """Opaque vendor-issued signature captured verbatim."""
MessageContentAttributes.MESSAGE_CONTENT_DATA      = "message_content.data"
    """Opaque vendor-issued data captured verbatim. Maps to Anthropic
       redacted_thinking.data."""
```

plus `llm.token_count.completion_details.reasoning` and
`llm.cost.completion_details.reasoning`. There is nothing to invent and no
bespoke `b2e.*` key needed for the reasoning payload itself. A `b2e.*` key here
would be a private name for a thing the ecosystem already names.

### An existing defect this exposes

Current `messages_api` `LLM` spans set `llm.input_messages` to a **JSON string**
via `telemetry.set_attr` (`sim/telemetry.py:192`). Read back out of Phoenix's
Postgres:

```json
"input_messages": "[{\"role\": \"user\", \"content\": \"Какие витрины доступны?…\"}]"
```

The convention defines these as *indexed flat keys* —
`llm.input_messages.0.message.role`, `.0.message.content` — which Phoenix
re-nests into an array on ingest and renders in its Messages panel. A JSON
string is stored as a string. Whether Phoenix renders it anyway is the one thing
in this document that has not been verified, which is why stage 0 exists.

---

## 6. Stages

### Stage 0 — settle the two open questions, before writing anything

Cheap, and both are the kind of question that silently invalidates an
implementation.

1. **Rendering.** Emit one hand-built `LLM` span to the live Phoenix twice: once
   with `llm.output_messages` as a JSON string, once with indexed flat keys
   including a `message_content.type="reasoning"` entry. Open both in the UI.
   Adopt whichever the Messages panel actually renders; if both render, adopt
   the indexed form, because it is what the convention defines and what any
   other tool reading this store will expect.
2. **Live-stream parity.** Run one turn with `--include-partial-messages` and
   keep the stream. Confirm whether assistant events carry `thinking` blocks
   live, and whether the partial-message events carry a usable first-token
   timestamp. This decides whether stage 4 is worth building and whether stage 1
   could be done live rather than post-hoc.

Both are read-only with respect to the bench: one turn, one arm, no config
change. **Effort:** an hour.

### Stage 1 — reasoning and completion content on the existing `LLM` spans

**This is the whole point of the document, and it is small.**

The spans already exist, are already parented to the turn, already populate
Phoenix's `llm_token_count_*` columns and already produce `span_costs` rows
(`34b09ef9…`, 11 spans, 11 cost rows). This stage gives them a body.

*How:*

- `LlmCall` (`claude_code.py:798`) gains `contents: list[dict]` and
  `tool_calls: list[dict]`.
- `parse_transcript` (`:846`) stops discarding repeat rows for a `message.id`
  and **merges their content blocks in row order** instead, keeping each block's
  own timestamp. Extend `ended_ns` to the last row of the message rather than
  the first — the current span closes before the message finished.
- `telemetry.py` gains one helper, `set_message_contents(span, key_prefix,
  role, blocks)`, that writes the indexed flat keys and is the only place that
  knows the layout. It imports `MessageAttributes` and
  `MessageContentAttributes` alongside the existing imports.
- `emit_llm_spans` (`:925`) calls it for `llm.output_messages.0`, mapping
  `thinking → message_content.type="reasoning"` with `.text` and `.signature`,
  `text → type="text"`, and `tool_use →
  llm.output_messages.0.message.tool_calls.N.tool_call.{id,function.name,function.arguments}`.
- Set `llm.token_count.completion_details.reasoning` where it can be derived;
  omit it where it cannot, rather than estimating.

*Acceptance:* on a real turn, every `LLM` span whose message contained a
`thinking` block shows the reasoning text in Phoenix's Messages panel; the
number of spans carrying reasoning equals the number of `thinking` blocks in the
transcript for that turn; and a turn whose transcript is missing still produces
`b2e.trace.llm_spans="missing"` and no invented content.

*Risks:* the reasoning-content path must be inside the same never-raises
discipline as the rest of `_read_transcript` — a telemetry defect may not cost
an answer. Add a `b2e.trace.llm_content` marker with the same vocabulary as
`b2e.trace.llm_spans` so a content-parsing failure is visible as a failure and
not as a model that did not reason.

**Effort:** small. A day including the test.

### Stage 2 — the conversation as `llm.input_messages`, labelled as partial

*How:* `parse_transcript` already walks the rows in order; accumulate the
conversation prefix as it goes and hand each `LlmCall` the messages that
preceded it. `emit_llm_spans` writes them as `llm.input_messages.N.message.*`,
with tool results as `message.role="tool"` and `message.tool_call_id`.

*The labelling is not optional.* Every such span carries:

- `b2e.llm.prompt_reconstruction = "conversation_only"`
- `b2e.llm.prompt_missing = "cli_system_prompt,tool_schemas"`
- `llm.system` set to the harness's own system prompt **plus** harness note —
  which is genuinely known — and `b2e.llm.system_partial = true`, because
  Claude Code's base prompt sits in front of it and is not here.

Anyone querying token cost against `llm.input_messages` on this harness must hit
these flags before they hit a wrong conclusion. This is the same discipline as
`b2e.llm.timing="derived"` and `b2e.trace.correlation="unmatched"`.

*Acceptance:* for a turn with N model calls, span *k* carries exactly the
messages that preceded call *k*, the last of which is the tool result the model
was responding to; and the reconstruction flags are present on every span.

**Effort:** small-to-medium. The ordering is fiddly when one message issues
parallel tool calls — the sample above does exactly that.

### Stage 3 — `CHAIN iteration.N`, the missing level of the tree

`docs/span-schema.md` lists this as absent on `claude_code` with the note
"nothing outside the CLI can observe" iteration boundaries. With the transcript
read in order, they are observable: one iteration is one `message.id` plus the
tool results that answered it.

*How:* in `emit_spans`, open a `CHAIN iteration.k` per model call, parent the
`LLM` span and its resulting `TOOL` spans under it, matching by `tool_use` id.
`ToolSpanRecorder` already holds `_finished` with the ids, and `claim_for`
already demonstrates the claim-once pattern.

*Acceptance:* the `claude_code` tree in `docs/span-schema.md` becomes the same
shape as the `messages_api` tree, and the doc's comparison table loses two
"**no**" cells.

*Risks:* re-parenting `TOOL` spans changes an established tree shape. Spans are
parented at creation, and `ToolSpanRecorder` creates them live from the drain
thread before the transcript is read — so either the iteration spans are opened
live from the stream (which requires stage 0's parity answer), or the tool spans
stay where they are and iteration spans wrap only the `LLM` span, which is a
weaker but honest version. Decide after stage 0. **Do not fabricate a parent.**

**Effort:** medium.

### Stage 4 — real per-call latency, if stage 0 says it is there

`--include-partial-messages` exists in 2.1.220 — confirmed in
`claude --help` inside the agent container: "Include partial message chunks as
they arrive (only works with --print and --output-format=stream-json)".

If those events carry a first-chunk timestamp, then time-to-first-token and true
per-call duration become measurements rather than derivations, and
`b2e.llm.timing` can say `"measured"` on turns that used the flag. That is the
one thing option 4b (the recording egress proxy, dropped) would have bought, at
none of its cost.

*Risks:* the flag increases stdout volume, and `on_event` runs on the drain
thread where a slow callback slows the turn. Gate it behind the same switch as
`_keeps_raw_stream` — experiment turns only — and fingerprint nothing, because
it changes no input to the model.

**Effort:** small if stage 0 is positive; otherwise do not build it.

### Stage 5 — not a tracing stage: make the prompt actually knowable

Listed because stage 2's tier-2 caveat will annoy someone into proposing it, and
it should be evaluated as what it is: **an experimental design change, not an
instrumentation one.**

2.1.220 offers `--system-prompt`, `--system-prompt-file` and
`--exclude-dynamic-system-prompt-sections`. Replacing Claude Code's base system
prompt with one this repo authors would make the request fully known, remove
~10 000 tokens of software-engineering instructions from a corporate HR agent's
context, and arguably make the bench *more* faithful to the thing it models — a
real B2E assistant does not carry Claude Code's prompt.

It would also change every arm's behaviour, invalidate comparison with every
trace recorded to date, and require a new `agent_config_version`. It belongs in
`docs/research-agenda.md`, behind a measurement, not here.

---

## 7. Size, retention, and the switch

Per the measurements in §1, one session's full content is roughly
208 KB / 16 ≈ **13 KB of reasoning** and ≈ **62 KB of total content** including
tool results. At 1 000 turns that is on the order of **60 MB** in Phoenix's
Postgres — against a store that currently holds 789 spans in total. Not a
concern; the `MAX_ATTR_BYTES = 128 KiB` guard in `sim/telemetry.py:45` already
covers the p100 `tool_result` of 50 KB with room to spare.

Latency: the transcript is read after the turn returns
(`claude_code.py:640`), off the answer path. Parsing content instead of only
usage adds a JSON walk over a file already being read. Nothing measurable.

**Retention needs no new decision, and this is worth stating because
`docs/subprocess-tracing-plan.md` raised it as a blocker for option 4b.** That
concern was specific to a TLS-terminating proxy recording raw requests. Here:
every byte is synthetic (README §Все данные синтетические), the tool results are
*already* on the `TOOL` spans in full, and the reasoning is the agent's own
words about synthetic people. The one real change is that reasoning text may
quote employee names and salaries from the corpus — which the `TOOL` span
beside it already contains verbatim.

Gate it anyway, on the existing distinction rather than a new one: `_keeps_raw_stream`
already separates experiment turns from conversation turns, and its docstring
already gives the reason ("ordinary conversation turns are not measurements").
Reasoning capture should follow the same switch, and turning it off must set
`b2e.trace.llm_content="disabled"` rather than leaving an absence.

---

## 8. What still stays unmeasurable

Correcting §7 of `docs/subprocess-tracing-plan.md` rather than repeating it:

- **Reasoning is now visible.** That sentence was wrong.
- **Per-token timing** remains unavailable. Stage 4 may reach first-token
  timing; nothing reaches the tokens between.
- **The exact request** remains unavailable without stage 5, and stage 5 is a
  different experiment. The ~10 000-token CLI prefix stays a labelled hole.
- **Why one query shape was chosen over another** is now *partially* visible —
  the model says why, in the reasoning, and that is evidence about its stated
  reasons rather than about its computation. Any attribute or analysis naming
  this must not blur the two.

The rule the earlier document argued for applies unchanged, and this plan is an
instance of it: **an instrumentation gap must be visible in the data, not only
in the code.** A `thinking` block that was read and discarded left no trace of
having existed. Had `parse_transcript` recorded "291 content blocks seen, 0
retained", the gap would have been obvious the first time anyone opened Phoenix.

---

## 9. Comparability

Each stage changes what a trace contains without changing what the agent does —
no stage here except 5 touches a fingerprint field. Per the standing rule:
traces recorded either side of a stage are not comparable *as observability*,
and the reference corpus should be re-run when a stage lands rather than
analysed across the boundary.

One additional note specific to this work: turns already run **can** be
retro-enriched, and uniquely so. The 16 transcripts measured in §1 are still on
the agent volume. A backfill that re-parses them and emits the reasoning onto
the existing traces is possible — but it would produce spans whose export time
is weeks after their start time, in a store where nothing else does that. Decide
explicitly; do not do it by accident.

---

## 10. Sequence

| Stage | What | State |
|---|---|---|
| 0.1 | verify Phoenix message rendering | **done — flat indexed keys win** |
| 0.2 | verify live-stream parity for `thinking` | **done — and it changed the answer** |
| 1 | reasoning + completion content on the existing `LLM` spans | **done, deployed** |
| 2 | conversation as `llm.input_messages`, flagged as partial | **done, deployed** |
| 3 | `CHAIN iteration.N` — the missing tree level | **done, deployed** |
| 4 | measured per-call time to first token | **done, deployed** |
| 5 | `--system-prompt` to make the request knowable | **not a tracing change** |

Stage 1 is independently shippable and carries most of the value. Nothing in
stages 0–4 changes a fingerprint field, changes what the model sees, or requires
the sessions to stop being stateless.

---

## 12. What actually shipped, 2026-08-09

### Stage 0.1 — answered

Two spans emitted to the deployed Phoenix (19.13.0), identical but for the
encoding. Read back from Postgres:

- **indexed flat keys** re-nest on ingest into
  `llm.output_messages[0].message.contents[…].message_content.{type,text,signature}`
  and `…tool_calls[0].tool_call.function.{name,arguments}` — structured, and
  what the UI renders.
- **the JSON-string form** is stored as a string and stays opaque.

So the flat form, and `sim/telemetry.set_messages` is the only place that knows
the layout. `record_llm_result` still writes the blob form on the
`messages_api` harness; that is a pre-existing defect deliberately left out of
this change's blast radius, since fixing it rewrites a corpus this work does not
touch.

### Stages 1 and 2 — built

| File | Change |
|---|---|
| `sim/telemetry.py` | `set_messages()`, plus `REASONING` / `TEXT` and the `MessageContentAttributes` import |
| `sim/agent/claude_code.py` | `LlmCall` gains `contents`, `tool_calls`, `input_messages`, `reasoning`; `_content_payload()`, `_user_messages()`, `_prompt_messages()`, `_answer_text()`, `capture_llm_content()` |
| `sim/agent/claude_code.py` | `parse_transcript` merges the rows of a message instead of dropping them, closes the window at the last row, and accumulates the conversation |
| `sim/agent/claude_code.py` | `ClaudeCodeResult.system_suffix`, so the spans can record the half of the system prompt that is known |
| `sim/agent/config.py` | the stale `--no-session-persistence` comment, and why it mattered |
| `docs/span-schema.md` | the `LLM` section, and two rows of the harness comparison table |
| `tests/test_sim_transcript_spans.py` | 12 tests |

Two defects were found and fixed on the way, neither of them the subject of this
work:

- `parse_transcript` closed a call's window at the **first** row of the message.
  On a model that reasons before it answers, that is the reasoning block — so
  every `LLM` span ended before the answer existed.
- `previous_ns` was not advanced for the dropped rows, so a call following a
  multi-row message with no user row between them started too early. Tool
  results usually hid it.

### Acceptance, run

- **Offline replay** of all 16 real transcripts: 291 model calls, **291 with
  non-empty reasoning**, 208 472 reasoning bytes, 308 tool calls — matching the
  independent census in §1 exactly, which is the check that the parser reads the
  file once and reads all of it.
- **End to end** on one real turn through the deployed exporter: 6 `LLM` spans,
  6 carrying reasoning, prompt prefixes growing 1 → 3 → 5 → 8 → 10 → 13, and
  `b2e.llm.prompt_reconstruction="conversation_only"` on every one. The recovered
  chain shows the agent noticing its own wasted call — *«Не очень помогло.
  Давайте просто получим нужные данные напрямую»* — which is exactly the RQ2
  event the bench existed to measure and had never been able to see.
- **Suite:** 425 → 437 tests, all passing.
- Both probe projects were deleted from Phoenix afterwards; `b2e-sim` is
  untouched at 789 spans.

### One deliberate deviation from this plan

§7 proposed gating content capture on `_keeps_raw_stream`, so only experiment
turns would carry reasoning. **That was wrong and was not implemented.** `TOOL`
spans already carry every tool result in full on every turn, and those hold far
more of the corpus than the agent's reasoning about it does — so the gate would
have protected nothing while blinding exactly the turns a researcher debugs, the
ones that went wrong in conversation.

Capture is therefore on by default, with `B2E_TRACE_LLM_CONTENT=0` to disable,
and disabling is recorded as `b2e.trace.llm_content="disabled"` rather than
leaving an absence. The reasoning is in `capture_llm_content()`.

---

## 13. Stage 0.2, and what it changed

Run live on 2026-08-09 against CLI 2.1.220 in the agent container: the same
question twice, once as the harness ran it and once with
`--include-partial-messages`. It answered more than it was asked.

**The live stream carries reasoning already, with or without the flag.**
`assistant` events contain `thinking` blocks, non-empty, exactly as the
transcript does. So the transcript was never the only route — it is simply the
route that also survives a crashed drain thread.

**`--include-partial-messages` adds a `ttft_ms` field.** The flag produces
`stream_event` rows — `message_start`, `content_block_start`,
`content_block_delta` (`thinking_delta`, `signature_delta`, `text_delta`),
`content_block_stop`, `message_delta`, `message_stop` — and the `message_start`
row carries `ttft_ms`, **a measured time to first token, keyed by `message.id`**.
That is the number §6 said only a recording egress proxy could buy, and option
4b was dropped rather than built for it. It was in the stream the whole time,
one flag away.

**The `result` envelope carries turn-level measurements nobody was reading**,
with or without the flag: `duration_api_ms`, `ttft_ms`, `ttft_stream_ms`,
`time_to_request_ms`.

**One thing that did *not* transfer.** The live `message_delta` usage carries
`output_tokens_details.thinking_tokens` — the reasoning token count. The 16
historic transcripts have no `output_tokens_details` at all (751 assistant rows,
zero occurrences), so `llm.token_count.completion_details.reasoning` is still
not set. Reading it from the stream is a future stage; it is deliberately not
estimated from character counts.

### So stages 3 and 4 were built too

| Stage | What landed |
|---|---|
| 3 | `ToolSpanRecorder` opens one `CHAIN iteration.N` per `message.id` seen on the stream, closes it when the next begins, and parents the turn's `TOOL` spans under it. `emit_llm_spans` files each `LLM` span under the iteration with the matching id — exact correlation, not a time heuristic, because the stream and the transcript name the same `msg_…`. |
| 4 | `--include-partial-messages` is now always passed; `ttft_ms` is captured per message and written as `b2e.llm.ttft_ms` with `b2e.llm.ttft_source="measured"`. The span *window* stays `derived` and still says so — one measured number and one derived number on the same span, each labelled. |
| 4 | The `result` timings land on the root as `b2e.turn.api_duration_ms` / `.ttft_ms` / `.ttft_stream_ms` / `.time_to_request_ms`, recorded and not reinterpreted. |

The iteration boundary is **the arrival of the previous iteration's last tool
result**, not first sight of the next message. The gap between them is the model
thinking, and it belongs to the call it produced; cutting at first sight would
put every `LLM` span fractionally before its own parent, because the transcript
dates a call from the row preceding it — which is exactly that tool result.

---

## 14. Live acceptance, and deployment

`docker compose build b2e-agent` + `up -d`, then one real turn through the
deployed API. `/healthz`: `tracing: exporting to http://phoenix:6006`,
`harness: ready (claude), proxy=yes`.

**Turn `d1f1df08addd90c79cea13ff9ab14e3e`** — 14 iterations, 13 tool calls, 10
Heimdall calls, 1 skill run, 244 960 tokens, $0.098.

Span census: 1 `AGENT`, **11 `CHAIN iteration.N`**, **11 `LLM`**, 13 `TOOL`, 10
`heimdall.*`. The tree came out as the `messages_api` tree — the shape this
harness was documented as unable to produce:

```
b2e.turn
└── iteration.7
    ├── llm.messages.create
    └── tool.mcp__heimdall__mcp_query
        └── heimdall.mcp_query
```

Every `LLM` span carried reasoning and a measured `ttft_ms` (824–4 387 ms), with
prompt prefixes growing 1 → 3 → 6 → 8 → 10 → 13 → 15 → 18 → 20 → 22 → 24.

And the reasoning earned its place immediately. The agent asked for headcount,
got `fact_count: 1`, and worked out why:

> «Ого, это странно. Результат для employee_actual показывает `fact_count: 1`…»
> «Интересно, has_next_page = false при limit 1, то есть всего только 1 сотрудник…»
> «Получается, что в моем доступе есть только 1 сотрудник — тот, к которому я привязан.»

Three iterations of an agent diagnosing its own row-level access restriction.
Before this change the trace recorded that it made three `mcp_query` calls and
returned a hedged answer, and nothing about why.

### Comparability

Per §9, this is a large observability change: the tree gained a level, `TOOL`
spans moved from the root to their iteration, and the `LLM` span window now
closes at the message's last row rather than its first. **Do not analyse across
2026-08-09.** Re-run the reference corpus.

---

## 11. Provenance

Following `docs/observability.md`: what was actually checked, how, and what was
not. This matters more than usual here, because the document's central claim
contradicts a sentence in an existing design doc.

### Verified on the live stack, 2026-08-09

| Claim | How |
|---|---|
| 291 `thinking` blocks across 16 B2E transcripts, all non-empty | script over `/app/var/claude-home/.claude/projects/*/*.jsonl` in `b2e-sim-b2e-agent-1` |
| all 16 transcripts `entrypoint: "sdk-cli"`, model `claude-haiku-4-5-20251001` | same census, 1 151 rows |
| byte distributions for thinking / text / tool_result | same census |
| headless persists reasoning, interactive redacts it | 409 transcripts on the dev host, 2 214 thinking blocks, split by `entrypoint` |
| one `message.id` spans several rows, one per content block, each timestamped | row-by-row dump of `1e4e0ec1-…` |
| `requestId` stable across a message's rows | same dump |
| ~12 200 prompt tokens on a 61-character first question | `in=10 cr=6526 cw=5690`, same dump |
| `attachment` rows carry only `deferred_tools_delta`, ~399 B | census across all 16 |
| semconv 0.1.31 defines `message_content.type="reasoning"`, `.signature`, `.data` | read out of the installed package in the agent container |
| existing `LLM` spans store `llm.input_messages` as a JSON string | `select jsonb_pretty(attributes) …` against Phoenix's Postgres |
| span census: `TOOL 520, LLM 112, CHAIN 97, AGENT 60` | same database |
| `--include-partial-messages`, `--system-prompt`, `--exclude-dynamic-system-prompt-sections` exist in 2.1.220 | `claude --help` in the agent container |
| container CLI is 2.1.220; dev host is 2.1.226 | `claude --version` in both |

### Not verified — do not treat these as settled

- **Whether Phoenix renders indexed flat message keys, or the JSON-string form,
  or both.** This is stage 0.1 and it decides the shape of stage 1's output.
- **Whether the live stream carries `thinking` blocks.** No kept
  `claude.stream.jsonl` existed on the volume to check — the 16 turns were not
  experiment turns. This is stage 0.2, and it decides whether stage 1 could run
  live instead of post-hoc and whether stage 4 is buildable.
- **Whether reasoning is present unconditionally, or depends on a thinking
  budget the harness does not set explicitly.** 16 sessions on one stack, all
  with reasoning, is strong but not conclusive; a configuration that suppressed
  it would look identical to a model that chose not to reason. Stage 1's
  `b2e.trace.llm_content` marker is what makes that distinguishable afterwards,
  and it should be treated as load-bearing rather than defensive.
- **Whether re-parenting `TOOL` spans under `CHAIN iteration.N` is feasible
  given they are created live from the drain thread.** Flagged in stage 3.

A multi-agent investigation was started to attack these independently and was
halted before its verification pass ran. The claims above stand on the direct
evidence cited, not on a second opinion.

### One documentation defect found along the way

`sim/agent/config.py`, in the `CONVERSATION_MODES` comment, still says of
`stateless`: "For ``claude_code`` that is ``--no-session-persistence``". That
flag was removed and the mechanism is now the absence of `--resume`, as
`sim/agent/claude_code.py:47` correctly records. The stale comment is what makes
"we must stay stateless, so we cannot have the transcript" look true. Worth
fixing regardless of whether any stage here is built.
