"""Build a trace export, and render it as the standalone explorer page.

One module, two callers with different sources and the same output:

* ``scripts/export_traces.py`` reads the backing Postgres, for an offline file;
* the admin UI's *Trace explorer* tab reads Phoenix's REST API, for a live page.

Keeping the derived fields, the analysis and the findings here is the point. The
two paths differed once already — the file said one thing about a corpus and the
UI would have said another about the same spans — and a viewer that disagrees
with the export it was built from is worse than no viewer.

The two sources do not agree on span shape, and neither is wrong:

* Postgres stores ``attributes`` as jsonb, already nested, with ``span_id`` on
  the row.
* The REST API returns attributes **flattened by dotted key**
  (``llm.output_messages.0.message.contents.0.message_content.type``) and hides
  the id under ``context.span_id``.

:func:`unflatten` is what reconciles them, and it has to rebuild *lists* as well
as objects: the message layout is an indexed sequence, and a viewer handed
``{"0": …, "1": …}`` where it expects an array renders nothing and says nothing.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

#: The explorer page. Lives under ``sim/`` so the Docker image carries it —
#: ``deploy/Dockerfile`` copies ``sim``, and a template at the repository root
#: would exist in the checkout and be missing in the container, which is the
#: kind of difference that only shows up in production.
TEMPLATE = Path(__file__).resolve().parent / "viewer" / "template.html"
PLACEHOLDER = "__DATA__"


# ------------------------------------------------------------------ reshaping


def unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Dotted keys to nested structures, with integer runs becoming lists.

    ``{"llm.input_messages.0.message.role": "user"}`` becomes
    ``{"llm": {"input_messages": [{"message": {"role": "user"}}]}}``.

    A node is turned into a list only when *every* key under it is an integer.
    Anything else stays an object, so an attribute that happens to be named
    ``0`` beside a named sibling does not silently reorder the tree.
    """
    root: dict[str, Any] = {}
    for key, value in (flat or {}).items():
        parts = str(key).split(".")
        node = root
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value
    return _listify(root)


def _listify(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    converted = {k: _listify(v) for k, v in node.items()}
    if converted and all(k.isdigit() for k in converted):
        return [converted[k] for k in sorted(converted, key=int)]
    return converted


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _utc(value: Any, *, millis: bool = False) -> str:
    stamp = _epoch(value)
    if not stamp:
        return ""
    moment = datetime.utcfromtimestamp(stamp)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if millis else \
        moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_rest_span(span: dict[str, Any]) -> dict[str, Any]:
    """One REST span in the shape the explorer and the export both expect."""
    context = span.get("context") or {}
    start, end = span.get("start_time"), span.get("end_time")
    return {
        "span_id": context.get("span_id") or span.get("span_id"),
        "parent_id": span.get("parent_id"),
        "name": span.get("name"),
        "span_kind": span.get("span_kind"),
        "start_utc": _utc(start, millis=True),
        "duration_ms": round((_epoch(end) - _epoch(start)) * 1000, 3),
        "status_code": span.get("status_code"),
        "status_message": span.get("status_message"),
        "attributes": unflatten(span.get("attributes") or {}),
        "events": span.get("events") or [],
        "_start_epoch": _epoch(start),
    }


# -------------------------------------------------------------- derived fields


def attr(span: dict[str, Any], *path: str) -> Any:
    node = span.get("attributes") or {}
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def enrich(trace: dict[str, Any]) -> dict[str, Any]:
    """Add the roll-ups the explorer reads off a trace, in place."""
    rows = trace.get("spans") or []
    root = next((s for s in rows if not s.get("parent_id")), None)
    trace["span_count"] = len(rows)
    if root:
        trace["user_prompt"] = attr(root, "input", "value")
        trace["final_answer"] = attr(root, "output", "value")
        trace["session_id"] = attr(root, "session", "id")
        trace["user_id"] = attr(root, "user", "id")
        trace["harness"] = attr(root, "b2e", "harness")
        trace["run_fingerprint"] = attr(root, "b2e", "run")
        trace["turn_stats"] = attr(root, "b2e", "turn")
        trace["turn_token_counts"] = attr(root, "llm", "token_count")
        trace["permission_denials"] = attr(root, "b2e", "permission_denials")
        trace["trace_markers"] = attr(root, "b2e", "trace")

    names: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for span in rows:
        names[span.get("name")] = names.get(span.get("name"), 0) + 1
        kinds[span.get("span_kind")] = kinds.get(span.get("span_kind"), 0) + 1
    trace["layer_counts"] = {
        "iteration_CHAIN": sum(v for k, v in names.items()
                               if str(k).startswith("iteration.")),
        "LLM": kinds.get("LLM", 0),
        "TOOL": kinds.get("TOOL", 0),
        "heimdall_CHAIN": sum(v for k, v in names.items()
                              if str(k).startswith("heimdall.")),
        "sandbox_CHAIN": names.get("sandbox.execute", 0),
    }
    trace["span_name_counts"] = names
    trace["reasoning"] = reasoning_summary(rows)
    return trace


def _reasoning_of(span: dict[str, Any]) -> list[str]:
    out = []
    for message in attr(span, "llm", "output_messages") or []:
        for item in ((message or {}).get("message") or {}).get("contents") or []:
            content = (item or {}).get("message_content") or {}
            if content.get("type") == "reasoning" and content.get("text"):
                out.append(content["text"])
    return out


def reasoning_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [s for s in rows if s.get("span_kind") == "LLM"]
    texts: list[str] = []
    with_reasoning = 0
    ttfts: list[int] = []
    for span in calls:
        mine = _reasoning_of(span)
        if mine:
            with_reasoning += 1
        texts.extend(mine)
        ttft = attr(span, "b2e", "llm", "ttft_ms")
        if ttft is not None:
            ttfts.append(int(ttft))
    return {
        "llm_spans": len(calls),
        "spans_with_reasoning": with_reasoning,
        "reasoning_blocks": len(texts),
        "reasoning_chars": sum(len(t) for t in texts),
        "ttft_ms": _spread(ttfts),
        "reasoning_texts": texts,
    }


def _spread(values: list[int]) -> dict[str, Any]:
    return {"n": len(values),
            "min": min(values) if values else None,
            "median": int(statistics.median(values)) if values else None,
            "max": max(values) if values else None}


# -------------------------------------------------------------------- analysis


def analyse(traces: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"iteration_CHAIN": 0, "LLM": 0, "TOOL": 0,
              "heimdall_CHAIN": 0, "sandbox_CHAIN": 0}
    for trace in traces:
        for key, value in (trace.get("layer_counts") or {}).items():
            totals[key] = totals.get(key, 0) + value

    reconciliation = []
    parenting = []
    ttfts: list[int] = []
    for trace in traces:
        turn = trace.get("turn_token_counts") or {}
        prompt = completion = 0
        misparented = []
        for span in trace.get("spans") or []:
            if span.get("span_kind") == "LLM":
                counts = attr(span, "llm", "token_count") or {}
                prompt += int(counts.get("prompt") or 0)
                completion += int(counts.get("completion") or 0)
            value = attr(span, "b2e", "llm", "ttft_ms")
            if value is not None:
                ttfts.append(int(value))
        by_id = {s.get("span_id"): s for s in trace.get("spans") or []}
        for span in trace.get("spans") or []:
            if span.get("span_kind") not in ("LLM", "TOOL"):
                continue
            parent = by_id.get(span.get("parent_id"))
            parent_name = parent.get("name") if parent else "(root)"
            if not str(parent_name).startswith("iteration."):
                misparented.append(f"{span.get('name')} under {parent_name}")
        reconciliation.append({
            "trace_id": trace.get("trace_id"),
            "turn_prompt": turn.get("prompt"),
            "llm_span_prompt_sum": prompt,
            "delta_prompt": (turn.get("prompt") or 0) - prompt,
            "turn_completion": turn.get("completion"),
            "llm_span_completion_sum": completion,
            "delta_completion": (turn.get("completion") or 0) - completion,
        })
        parenting.append({"trace_id": trace.get("trace_id"),
                          "spans_not_under_an_iteration": misparented})

    started = [t.get("started_utc") for t in traces if t.get("started_utc")]
    return {
        "corpus": {
            "traces": len(traces),
            "spans": sum(t.get("span_count") or 0 for t in traces),
            "window_utc": [min(started), max(started)] if started else [],
        },
        "layer_totals": totals,
        "token_reconciliation": reconciliation,
        "parenting": parenting,
        "reasoning": {
            "llm_spans": sum(t["reasoning"]["llm_spans"] for t in traces),
            "spans_with_reasoning": sum(t["reasoning"]["spans_with_reasoning"]
                                        for t in traces),
            "reasoning_blocks": sum(t["reasoning"]["reasoning_blocks"]
                                    for t in traces),
            "reasoning_chars": sum(t["reasoning"]["reasoning_chars"]
                                   for t in traces),
        },
        "ttft_ms": _spread(ttfts),
    }


#: Bash commands that read the filesystem. The tool policy denies ``Read``,
#: ``Glob`` and ``Grep`` outright — ``sim/agent/claude_code.py`` says why: the
#: corpus is a directory of files and one of them is the answer key. Whether the
#: ``Bash`` form is equally denied is a question about the CLI's matcher, not
#: about our allowlist, so it is asked of the data rather than assumed.
FILE_READING_COMMANDS = ("grep", "cat", "head", "tail", "less", "awk", "sed",
                         "find", "ls ", "od ", "xxd", "strings")

#: How a refusal reads. Two wordings, because the harness changed permission
#: mode: ``dontAsk`` states the refusal as final, and the default mode before it
#: described a pending decision. A corpus spans both, and matching only the
#: current wording would score every older refusal as a successful read.
DENIAL_MARKERS = (
    "Permission to use Bash has been denied",           # permissionMode=dontAsk
    "requested permissions to use",                     # the earlier default
    "haven't granted it yet",
)


def bash_file_reads(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every Bash call that tried to read a file, and how it ended.

    Three outcomes, not two. A span with no recorded output cannot be scored:
    tool results were not captured on this harness until 2026-08-08, so an empty
    output means *the trace does not say*, and calling that a successful read
    would manufacture a finding out of an instrumentation gap — which is the
    error this whole schema exists to avoid.
    """
    found = []
    for trace in traces:
        for span in trace.get("spans") or []:
            if span.get("name") != "tool.Bash":
                continue
            raw = attr(span, "input", "value") or ""
            try:
                command = json.loads(raw).get("command", "")
            except (ValueError, TypeError, AttributeError):
                command = str(raw)
            command = str(command).strip()
            if not any(command.startswith(c) for c in FILE_READING_COMMANDS):
                continue
            output = str(attr(span, "output", "value") or "")
            if any(m in output for m in DENIAL_MARKERS):
                outcome = "denied"
            elif not output.strip():
                outcome = "unrecorded"
            else:
                outcome = "read"
            found.append({
                "trace_id": trace.get("trace_id"),
                "command_head": command[:200],
                "binary": command.split()[0] if command.split() else "",
                "outcome": outcome,
                "denied": outcome == "denied",
                "output_bytes": len(output),
            })
    return found


def findings(traces: list[dict[str, Any]],
             analysis: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    layers = analysis["layer_totals"]
    if layers["LLM"]:
        agree = layers["iteration_CHAIN"] == layers["LLM"]
        out.append({
            "id": "iteration-level-exists",
            "kind": "verification" if agree else "anomaly",
            "statement": f"{layers['iteration_CHAIN']} iteration.N spans against "
                         f"{layers['LLM']} LLM spans across "
                         f"{analysis['corpus']['traces']} traces"
                         + ("." if agree else " — these should be equal."),
            "evidence": "_analysis.layer_totals",
            "reading": "docs/span-schema.md called this level unobservable "
                       "outside the CLI. It is observable: the stream names the "
                       "message, and one message id is one iteration.",
        })
    bad = [p for p in analysis["parenting"] if p["spans_not_under_an_iteration"]]
    out.append({
        "id": "tree-is-fully-parented",
        "kind": "verification" if not bad else "anomaly",
        "statement": f"{len(traces) - len(bad)}/{len(traces)} traces have every "
                     f"LLM and TOOL span filed under an iteration.",
        "evidence": "_analysis.parenting",
        "reading": "Correlation is by message id, which the stream and the "
                   "transcript both name — exact, not a time heuristic. Traces "
                   "recorded before 2026-08-09 have no iteration layer at all "
                   "and will show up here.",
    })
    off = [r for r in analysis["token_reconciliation"]
           if r["delta_prompt"] or r["delta_completion"]]
    out.append({
        "id": "llm-spans-reconcile-exactly",
        "kind": "verification" if not off else "anomaly",
        "statement": f"Per-call tokens sum exactly to the turn totals on "
                     f"{len(traces) - len(off)}/{len(traces)} traces.",
        "evidence": "_analysis.token_reconciliation",
        "reading": "The transcript is read once, not twice, and no call is "
                   "dropped.",
    })
    reasoning = analysis["reasoning"]
    if reasoning["llm_spans"]:
        out.append({
            "id": "reasoning-is-present",
            "kind": "verification",
            "statement": f"{reasoning['spans_with_reasoning']}/"
                         f"{reasoning['llm_spans']} LLM spans carry non-empty "
                         f"reasoning ({reasoning['reasoning_chars']} chars).",
            "evidence": "_analysis.reasoning",
            "reading": "Headless sessions persist reasoning in full. Before "
                       "2026-08-09 all of this was parsed and discarded.",
        })
    reads = bash_file_reads(traces)
    got = [r for r in reads if r["outcome"] == "read"]
    unknown = [r for r in reads if r["outcome"] == "unrecorded"]
    if got:
        out.append({
            "id": "bash-file-read-bypasses-the-tool-policy",
            "kind": "defect",
            "statement": f"{len(got)} Bash calls read a file and got content "
                         f"back, using "
                         f"{sorted({r['binary'] for r in got})}. The tool policy "
                         f"denies Read/Glob/Grep precisely so the agent has no "
                         f"file reader."
                         + (f" A further {len(unknown)} such calls have no "
                            f"recorded output and cannot be scored either way."
                            if unknown else ""),
            "evidence": "_findings_detail.bash_file_reads",
            "reading": "The matcher permits some read-only Bash commands "
                       "regardless of --allowed-tools; echo/pwd/ls/whoami were "
                       "probed and these were not. The corpus is not mounted "
                       "into the agent, so the answer key stays out of reach — "
                       "but the agent can re-read its own session transcript "
                       "and any tool result the CLI spilled to disk, which is a "
                       "way round its own context window and a confound for any "
                       "question about context strategy or token cost.",
        })
    return out


def build_document(traces: list[dict[str, Any]],
                   about: dict[str, Any]) -> dict[str, Any]:
    analysis = analyse(traces)
    return {
        "_about": about,
        "_analysis": analysis,
        "_findings": findings(traces, analysis),
        "_findings_detail": {"bash_file_reads": bash_file_reads(traces)},
        "traces": traces,
    }


# --------------------------------------------------------------- live source

#: What a reconstructed prompt costs to carry. ``llm.input_messages`` is the
#: conversation up to each call, so a turn of N calls repeats it N times —
#: quadratic, and on a long turn it is most of the payload. It is a labelled
#: reconstruction whose content is already visible span by span, so the live
#: page drops it by default and says how many messages it dropped. The offline
#: export keeps everything.
def strip_input_messages(traces: list[dict[str, Any]]) -> int:
    dropped = 0
    for trace in traces:
        for span in trace.get("spans") or []:
            llm = (span.get("attributes") or {}).get("llm")
            if isinstance(llm, dict) and isinstance(llm.get("input_messages"), list):
                dropped += len(llm["input_messages"])
                llm["input_messages_dropped_from_view"] = len(llm.pop("input_messages"))
    return dropped


def from_phoenix(client: Any, *, limit: int = 100, project: str = "b2e-sim",
                 source: str = "", include_input_messages: bool = False,
                 exported_at: str = "") -> dict[str, Any]:
    """The explorer document for the newest ``limit`` traces in Phoenix."""
    raw = client.latest_traces(limit=limit)
    traces = []
    for entry in raw:
        spans = [normalise_rest_span(s) for s in entry.get("spans") or []]
        spans.sort(key=lambda s: s.get("_start_epoch") or 0)
        for span in spans:
            span.pop("_start_epoch", None)
        trace = {
            "trace_id": entry.get("trace_id"),
            "started_utc": _utc(entry.get("start_time")),
            "duration_s": round(_epoch(entry.get("end_time"))
                                - _epoch(entry.get("start_time")), 3),
            "spans": spans,
        }
        traces.append(enrich(trace))
    # Newest first in the list, which is the order a researcher wants when the
    # question is "what just happened".
    traces.sort(key=lambda t: t.get("started_utc") or "", reverse=True)

    dropped = 0 if include_input_messages else strip_input_messages(traces)
    about = {
        "source": source or f"Arize Phoenix REST, project '{project}'",
        "exported_at_utc": exported_at,
        "selection": f"the {len(traces)} most recent traces by start time, "
                     f"all spans, live from Phoenix.",
        "format": "the span tree in docs/span-schema.md for harness=claude_code: "
                  "AGENT root, CHAIN iteration.N per model call, LLM under its "
                  "iteration carrying reasoning and a measured ttft, TOOL under "
                  "the iteration that asked for it, heimdall.* and "
                  "sandbox.execute under their tool call.",
        "caveats": [
            "b2e.llm.timing='derived': the LLM span window comes from two "
            "transcript timestamps, nothing timed the request. b2e.llm.ttft_ms "
            "is the one measured latency.",
            "b2e.turn.api_duration_ms is the CLI's own figure and is not a "
            "wall-clock slice of the turn.",
            "Traces recorded before 2026-08-09 have no iteration layer and no "
            "reasoning; they are not comparable as observability.",
        ],
    }
    if dropped:
        about["caveats"].insert(0, _INPUT_MESSAGES_NOTE.format(dropped=dropped))
    return build_document(traces, about)


_INPUT_MESSAGES_NOTE = (
    "llm.input_messages is omitted from this page ({dropped} messages across "
    "the corpus). It is a labelled reconstruction of the conversation, repeated "
    "in full on every call of a turn, so it dominates the payload while showing "
    "nothing the individual spans do not. Use scripts/export_traces.py for a "
    "file that keeps it."
)


# ------------------------------------------------------------------ rendering


def render(document: dict[str, Any], template: str | None = None) -> str:
    """Embed the export in the explorer page.

    ``</`` is escaped so a payload containing ``</script>`` cannot close the
    embedding tag. ``<\\/`` is a legal JSON escape, so the data is unchanged —
    and the payload here is agent output, which is exactly the input nobody
    should trust.
    """
    if template is None:
        template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(document, ensure_ascii=False,
                         separators=(",", ":"), default=str).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)
