# Tracing the work that happens in the subprocess

**Status:** stages 1, 2 and 3 landed 2026-08-08. Stage 4a was run and settled the
question it existed to ask; 4b remains a proposal, blocked on a decision that is
not an engineering one.

---

## 1. The problem

On `harness=claude_code` — the default, and the harness behind 48 of the first
50 traces — the agent's real work happens inside a `claude -p` subprocess. The
model call goes from that subprocess straight to Anthropic. The Heimdall HTTP
request goes from a *further* subprocess, the stdio MCP bridge the CLI spawns.
Neither ever enters this process.

Everything in `sim/telemetry.py` instruments this process. It was written for
`sim/agent/loop.py`, where the loop, the model call and the HTTP call all run
here — and on that path it produces exactly the tree `docs/span-schema.md`
describes. On the default path it had nothing to instrument, so what it recorded
instead was a reconstruction assembled after the subprocess exited.

What that cost, measured on the 2026-08-01..05 corpus (50 traces, 447 spans):

| Symptom | Consequence |
|---|---|
| every `TOOL` span 0.000 s | an 81 s average turn could not be attributed to anything |
| 2 `LLM` spans corpus-wide (both from `loop.py`) | no per-call prompts, completions or finish reasons |
| all `llm_token_count_*` NULL | Phoenix's token and cost columns empty for every turn |
| no Heimdall `CHAIN` spans | tool-call-to-HTTP ratio unverifiable from the trace |
| no sandbox spans | `record_skill_execution` exists and is called by nobody |

The last one is worth stating plainly: `sim/telemetry.py:348` has been dead code
since it was written.

---

## 2. What stage 1 already fixed, and its ceiling

Landed:

- `ToolSpanRecorder` opens a `TOOL` span when the CLI announces a `tool_use` and
  closes it when the matching `tool_result` arrives. Durations are now real,
  measured at the two moments the stream reports them.
- Spans are parented explicitly through `telemetry.open_tool_span(parent=...)`,
  because `on_event` runs on the stdout-draining thread and OTel's current-span
  context is thread-local. Without that they would land in their own trace.
- The root span carries the turn's token totals under the standard
  `llm.token_count.*` keys, with the cache split beside them, so Phoenix's own
  columns populate.
- `b2e.turn.tool_time_ms` makes model time derivable as
  `duration_ms - tool_time_ms`.
- Experiment turns keep their raw stream (`_keeps_raw_stream`), so a future
  instrumentation change can be validated against turns already run.

**The ceiling.** Stage 1 measures the *boundaries* of each tool call as seen
from outside. It cannot see: how much of a tool call was HTTP versus MCP
overhead, what the model was doing between calls beyond "not in a tool", how
many tokens any individual call cost, or what happened inside the skill sandbox.
Those need spans emitted from where the work is.

---

## 3. What we actually control in there

This is the part that decides which stages are cheap and which are research
projects.

| Component | Ours? | Reachable |
|---|---|---|
| `heimdall/bridge.py` — the stdio MCP server | **yes**, our file, our env | fully |
| `/opt/skills/run` + sandbox worker | **yes** | fully |
| `claude` CLI | no — a binary we invoke | only via its stream, env and flags |
| the model call itself | no — CLI to Anthropic over the egress proxy | only at the network edge |

Two of the four gaps sit behind code we own. Those are stages 2 and 3, and they
are small. The model call is genuinely outside, and that is stage 4.

---

## 4. Stage 2 — Heimdall `CHAIN` spans from the bridge

**This is the highest value per unit of work, and most of it already exists.**

`heimdall/bridge.py:295` already has `_log()`, which records per call: tool
name, argument keys, HTTP status, `duration_ms` measured around the request,
error code, row count and payload bytes. It writes JSONL to `HR_TRACE_LOG` and
was built precisely as "источник трейса, не зависящий от того, что скажет агент".

It is never switched on. `ClaudeCodeHarness.mcp_config()` does not set
`HR_TRACE_LOG`, so the file is never written.

### Design

1. `mcp_config()` sets `HR_TRACE_LOG` to a per-turn path in the session workdir.
2. `mcp_config()` also sets `TRACEPARENT` to the W3C encoding of the root span,
   built from `telemetry.current_trace_id()` / `current_span_id()`. The root
   span is already open when `harness.run()` is called, so it is available.
3. `_log()` gains three fields: the wall-clock start time, the traceparent it
   was given, and the request path. Nothing else changes — it stays inside the
   stdlib.
4. After the turn, `emit_spans` reads the JSONL and materialises one `CHAIN`
   span per line, with `start_time`/`end_time` taken from the record.

Point 4 is a replay, which is what stage 1 removed — but here it is legitimate,
and the distinction matters: the timing is *measured in the bridge, at the call*,
and merely transported through the file. Stage 1's problem was not that spans
were created late, it was that nothing had measured them at all.

### Why not just run OpenTelemetry in the bridge

Because the bridge is stdlib-only by construction, and that constraint is
load-bearing: run hermeticity depends on `manifest_hash` over the hashes of
mounted files, and there is no virtualenv in `RUN_DIR`. An `import
opentelemetry` there fails mid-run, not at build. A hand-rolled OTLP writer
would technically fit in the stdlib but adds a protocol implementation to the
one file that must never surprise anyone.

### Caveats to verify before building

- **One bridge process per turn?** The design assumes each `claude -p`
  invocation spawns a fresh MCP server, so a per-turn `TRACEPARENT` in its
  environment is correct. This holds for stateless turns by construction.
  For `conversation_mode=resume` it must be *checked*, not assumed — if the CLI
  ever reuses a server across turns, the env is stale and every span after the
  first would be filed under the wrong trace. Test: run two turns in one
  resumed session, assert two distinct `TRACEPARENT` values appear in the log.
- **The log must be per turn**, not per session workdir, or a resumed session
  accumulates and the second turn re-emits the first turn's calls.
- Bridge failures must stay silent, as `_log` already is. Telemetry may not be
  able to break a turn.

### Acceptance

For a turn with N Heimdall tool calls, the trace shows N `CHAIN` spans nested
under their `TOOL` spans, each carrying `b2e.http.status`, `b2e.heimdall.rows`
and a non-zero duration; and the count matches the emulator's access log for
the same window. That last check is the one that would have caught this class of
bug in the first place.

**Effort:** small. Roughly a day including the resume verification.

---

## 5. Stage 3 — sandbox and skill spans

`telemetry.record_skill_execution()` already defines the contract
(`b2e.skill.hash`, `b2e.skill.state`, `b2e.sandbox.exit_status`,
`b2e.sandbox.wall_ms`, `b2e.sandbox.peak_rss_kb`,
`b2e.sandbox.rejected_imports`) and `docs/span-schema.md` documents it. Nothing
calls it, so on the default harness an approved-skill execution appears in the
trace as `tool.Bash` and nothing else — the digest that actually ran is not in
the trace, which defeats the point of recording the digest.

Same mechanism as stage 2: the runner writes a sidecar JSON per execution into
the turn's workdir, `emit_spans` materialises the span under the owning
`tool.Bash`. The runner is ours and is not stdlib-constrained, so if it turns
out to be simpler to emit OTLP directly from the sandbox worker, that is open —
but the sidecar keeps the sandbox free of network egress, which is a property
worth more than the elegance.

**Effort:** small, and it makes `b2e.skill.hash` mean what the threat model
already claims it means.

---

## 6. Stage 4 — per-call model visibility

The genuinely hard one. Three options, none free.

### 4a. The CLI's own telemetry export — **run 2026-08-08, answered: metrics only**

The installed CLI is **2.1.220**. It does support OpenTelemetry, and the binary
contains the full set of standard variable names, `OTEL_TRACES_EXPORTER` and
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` among them:

```
docker exec b2e-sim-b2e-agent-1 sh -c \
  'grep -aoE "OTEL_[A-Z_]{4,40}|CLAUDE_CODE_ENABLE_TELEMETRY" \
   /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe | sort -u'
```

**Their presence proves nothing** — they are what a bundled OTel SDK parses,
not a statement about what this CLI emits. The probe that settles it is a real
turn with the exporter pointed at Phoenix:

```
CLAUDE_CODE_ENABLE_TELEMETRY=1 OTEL_METRICS_EXPORTER=otlp OTEL_TRACES_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006 \
claude -p "скажи: ок" --model claude-haiku-4-5-20251001 --output-format json
```

Result, read off Phoenix's access log:

```
INFO: 172.19.0.2:40312 - "POST /v1/metrics HTTP/1.1" 405 Method Not Allowed
```

One `POST /v1/metrics`, rejected by Phoenix because Phoenix stores traces. **Zero
`POST /v1/traces`** — including on a second run with
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` set explicitly, where nothing was sent at
all. Phoenix's span store was unchanged after both.

So: this CLI version emits **metrics, not spans**, for an agent turn. That
settles 4a. Per-call `LLM` spans cannot be had this way, and pointing the CLI at
Phoenix accomplishes nothing.

What it does leave on the table: the metrics stream may well carry per-request
token counts, and consuming it would need a metrics sink — a Prometheus or an
OTel collector — beside Phoenix, not inside it. That is a separate piece of
infrastructure for a number the root span now already reports at turn
granularity. Not worth it yet.

Two further constraints if anyone revisits this: metrics correlate to a turn
only by time and session, never natively; and `build_argv` passes
`--setting-sources ""` deliberately, so any configuration path running through
settings files is off by design and would have to be re-admitted explicitly and
fingerprinted.

### 4b. A recording proxy at the egress

Every outbound call already goes through one place: `AGENT_HTTP_PROXY` →
`proxy-relay` → the host proxy. A recording proxy for `api.anthropic.com` would
see every request and response, including `usage`, and could emit a proper `LLM`
span per call with prompts, completions and token counts — the full tree the
schema doc describes.

Cost, stated honestly:

- It must terminate TLS to see bodies, which means a CA in the agent image.
  That is a real change to the security posture of the stand, even a synthetic
  one, and needs a decision rather than an implementation.
- Correlating a request to a turn needs a key the CLI does not send. Time plus
  the fact that turns are serialised per session gets most of the way; it is
  heuristic, and a heuristic correlation must be labelled as one in the span.
- It records full prompts, which on this stack contain synthetic personal data.
  Retention has to be decided up front.

### 4c. Claude Code hooks

`PreToolUse` / `PostToolUse` hooks could emit spans from inside the session with
the CLI's own view of a call. This duplicates what stage 1 already gets from the
stream, does not reach the model call at all, and conflicts with
`--setting-sources ""`. Listed for completeness; not recommended.

### 4c. The CLI's session transcript — **built 2026-08-08, and this is the answer**

Neither 4a nor 4b, and better than both. Claude Code writes a JSONL transcript
of each session under `<claude_home>/.claude/projects/<project>/<session>.jsonl`,
and every assistant row in it carries `message.usage`. Grouped by `message.id`,
those rows *are* the API calls.

Measured on a real transcript before anything was written: 77 rows, 43 of them
assistant, **15 distinct `message.id`** — 15 model calls, each with
`input_tokens`, `output_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens`, `model` and `stop_reason`.

What made it usable was one flag. `--no-session-persistence` suppressed the
file, and it was passed on every stateless turn — which is every experiment
turn. Two checks settled whether it could be dropped:

- **Parity.** Same question, with and without the flag: identical tool sequence
  (`ToolSearch` → `list_models`), identical `turns=3, tools=2, heimdall=1`, same
  answer.
- **Context leak.** Persistence on, *same working directory*, no `--resume`, two
  turns: the second had no memory of the first and said so. **What isolates a
  turn is the absence of `--resume`, not the absence of a file.**

Both passed, so the flag is gone. What isolates experiment arms is now a single
mechanism, which makes `test_a_stateless_config_never_resumes` load-bearing.

Verified live afterwards (trace `34b09ef9…`): 11 `LLM` spans on an 11-iteration
turn, per-call prompt tokens summing exactly to the turn total, Phoenix's own
token columns populated and 11 `span_costs` rows produced.

Two limits, both stated on the spans themselves:

- **No latency.** The transcript has no duration field of any kind — checked for
  `ttft`, `duration`, `latency`, `elapsed`, `_ms`; all zero, `diagnostics` null.
  Durations are derived from the gap between recorded timestamps and every span
  carries `b2e.llm.timing="derived"`.
- **No request.** The conversation is there, the system prompt and tool schemas
  are not. So no per-call prompt text — which is also why this route needs no
  retention decision.

The format is undocumented and tied to CLI 2.1.220, so `parse_transcript` raises
`TranscriptFormatError` when a transcript has assistant rows but none it can
read, and the turn is marked `b2e.trace.llm_spans="format"`. A parser that
silently returned nothing would recreate the original defect exactly: an absent
measurement that reads as a measured zero.

### Recommendation

**4b is dropped.** 4c gets everything the proxy would have given except true
per-call latency and the request body, at no infrastructure cost, with exact
correlation instead of a time heuristic, and without a CA in the agent image or
full prompts on disk.

Build 4b only if a question turns up that specifically needs real per-call
latency or the exact bytes sent. None currently does.

---

## 7. What stays unmeasurable, and saying so

> **Corrected 2026-08-09.** The paragraph below was wrong about reasoning, and
> the correction is `docs/reasoning-tracing-plan.md`. The reasoning was in the
> transcript this document had just taught the stack to read — 291 `thinking`
> blocks across the 16 sessions then on the deployed volume, every one of them
> non-empty — and `parse_transcript` was taking `message.usage` off those rows
> and discarding `message.content`. What was true is narrower: the trace does
> not see the *request*, and it does not see per-token timing. It now sees what
> the model said, because the CLI wrote it down.

Even with all four stages, the trace never sees inside the model: no reasoning,
no per-token timing, no view of why one query shape was chosen over another.
Nothing planned here changes that, and no attribute should ever be named in a
way that suggests otherwise.

The rule this whole episode argues for: **an instrumentation gap must be visible
in the data, not only in the code.** A tool span with a 0.000 s duration looked
like a fast tool. Had it been absent, or flagged, the gap would have been
obvious the first time anyone opened Phoenix instead of surviving 50 traces.
Where a stage below cannot be completed, the affected span should carry an
explicit marker — as `b2e.tool.unfinished` now does — rather than a plausible
zero.

---

## 8. Sequence

| Stage | What | State |
|---|---|---|
| 1 | live tool spans, turn tokens, kept streams | **done** |
| 2 | Heimdall `CHAIN` spans from the bridge log | **done** |
| 3 | sandbox / skill spans | **done** |
| 4a | verify what CLI 2.1.220 exports | **done — metrics only, no spans** |
| 4b | recording egress proxy for `LLM` spans | **dropped** — superseded by 4c |
| 4c | `LLM` spans from the CLI session transcript | **done** |

Every layer the schema describes now exists on the default harness except
per-iteration `CHAIN` spans, which nothing outside the CLI can observe.

Stage 3 landed cheaper than planned, and the reason is worth recording. The plan
above proposed a sidecar file from the runner. Reading `deploy/skills-run` showed
that unnecessary: the runner already prints digest, state, wall time, peak RSS
and rejected imports as the JSON on its stdout, which *is* the Bash tool result
the recorder already sees. The measurements were crossing the process boundary
the whole time and being discarded on arrival. No new channel was built.

One correction to the plan's own reasoning, since it was written before the code
was read: it claimed `record_skill_execution` needed a sidecar to be reachable.
It needed a caller.

One cross-cutting note on comparability: each stage changes what a trace
contains, so traces recorded either side of a stage are not comparable *as
observability*, even though the agent's behaviour is untouched — no stage here
alters a fingerprint field. When a stage lands, re-run the reference corpus
rather than analysing across the boundary.
