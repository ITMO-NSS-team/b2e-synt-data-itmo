"""Export Phoenix traces to a self-contained JSON file, with analysis.

Reads the backing Postgres directly rather than Phoenix's REST API: the API
paginates and reshapes, and the point of an export is to be the same bytes the
store holds. Nothing is sampled and no payload is truncated.

    python3 scripts/export_traces.py --since-rowid 61 -o var/phoenix-traces.json

The shape matches `var/phoenix-traces-2026-08-08.json`: `_about` says where it
came from, `_analysis` holds the cross-trace figures, `_findings` states what
those figures show, and `traces` carries every span in full.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from typing import Any

CONTAINER = "b2e-sim-postgres-1"
DB_USER = "phoenix"
DB_NAME = "phoenix"


def query(sql: str) -> list[dict[str, Any]]:
    """One query, returned as rows of dicts. Postgres does the JSON."""
    wrapped = (f"select coalesce(json_agg(t), '[]'::json)::text "
               f"from ({sql}) t")
    out = subprocess.run(
        ["docker", "exec", CONTAINER, "psql", "-U", DB_USER, "-d", DB_NAME,
         "-t", "-A", "-c", wrapped],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip() or "[]")


def attr(span: dict[str, Any], *path: str) -> Any:
    node = span.get("attributes") or {}
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def build(since_rowid: int, project: str) -> dict[str, Any]:
    traces = query(f"""
        select t.id as phoenix_rowid, t.trace_id,
               to_char(t.start_time at time zone 'UTC',
                       'YYYY-MM-DD"T"HH24:MI:SS"Z"') as started_utc,
               extract(epoch from (t.end_time - t.start_time)) as duration_s
        from traces t join projects p on t.project_rowid = p.id
        where p.name = '{project}' and t.id >= {since_rowid}
        order by t.id
    """)
    if not traces:
        return {}

    ids = ", ".join(str(t["phoenix_rowid"]) for t in traces)
    spans = query(f"""
        select s.trace_rowid, s.span_id, s.parent_id, s.name, s.span_kind,
               to_char(s.start_time at time zone 'UTC',
                       'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"') as start_utc,
               extract(epoch from (s.end_time - s.start_time)) * 1000 as duration_ms,
               s.status_code, s.status_message, s.attributes, s.events
        from spans s where s.trace_rowid in ({ids})
        order by s.trace_rowid, s.start_time
    """)
    costs = query(f"""
        select trace_rowid, sum(total_cost) as cost
        from span_costs where trace_rowid in ({ids}) group by trace_rowid
    """)
    cost_by_trace = {c["trace_rowid"]: float(c["cost"] or 0) for c in costs}

    by_trace: dict[int, list[dict]] = {}
    for span in spans:
        by_trace.setdefault(span["trace_rowid"], []).append(span)

    for trace in traces:
        rows = by_trace.get(trace["phoenix_rowid"], [])
        root = next((s for s in rows if s["parent_id"] is None), None)
        trace["span_count"] = len(rows)
        trace["span_cost_usd_sum"] = round(cost_by_trace.get(
            trace["phoenix_rowid"], 0.0), 6)
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

        kinds: dict[str, int] = {}
        names: dict[str, int] = {}
        for span in rows:
            kinds[span["span_kind"]] = kinds.get(span["span_kind"], 0) + 1
            names[span["name"]] = names.get(span["name"], 0) + 1
        trace["layer_counts"] = {
            "iteration_CHAIN": sum(v for k, v in names.items()
                                   if k.startswith("iteration.")),
            "LLM": kinds.get("LLM", 0),
            "TOOL": kinds.get("TOOL", 0),
            "heimdall_CHAIN": sum(v for k, v in names.items()
                                  if k.startswith("heimdall.")),
            "sandbox_CHAIN": names.get("sandbox.execute", 0),
        }
        trace["span_name_counts"] = names
        trace["reasoning"] = _reasoning_summary(rows)
        trace["spans"] = [{k: v for k, v in s.items() if k != "trace_rowid"}
                          for s in rows]
    return {"traces": traces}


def _reasoning_summary(rows: list[dict]) -> dict[str, Any]:
    """What the new layer actually recovered on this turn."""
    calls = [s for s in rows if s["span_kind"] == "LLM"]
    texts, ttfts = [], []
    for span in calls:
        messages = attr(span, "llm", "output_messages") or []
        for message in messages:
            for item in (message.get("message") or {}).get("contents") or []:
                content = item.get("message_content") or {}
                if content.get("type") == "reasoning" and content.get("text"):
                    texts.append(content["text"])
        ttft = attr(span, "b2e", "llm", "ttft_ms")
        if ttft is not None:
            ttfts.append(int(ttft))
    return {
        "llm_spans": len(calls),
        "spans_with_reasoning": sum(
            1 for s in calls
            if any((i.get("message_content") or {}).get("type") == "reasoning"
                   and (i.get("message_content") or {}).get("text")
                   for m in (attr(s, "llm", "output_messages") or [])
                   for i in (m.get("message") or {}).get("contents") or [])),
        "reasoning_blocks": len(texts),
        "reasoning_chars": sum(len(t) for t in texts),
        "ttft_ms": {"n": len(ttfts),
                    "min": min(ttfts) if ttfts else None,
                    "median": int(statistics.median(ttfts)) if ttfts else None,
                    "max": max(ttfts) if ttfts else None},
        "reasoning_texts": texts,
    }


def analyse(traces: list[dict]) -> dict[str, Any]:
    totals = {"iteration_CHAIN": 0, "LLM": 0, "TOOL": 0,
              "heimdall_CHAIN": 0, "sandbox_CHAIN": 0}
    for trace in traces:
        for key, value in (trace.get("layer_counts") or {}).items():
            totals[key] = totals.get(key, 0) + value

    reconciliation = []
    for trace in traces:
        turn = trace.get("turn_token_counts") or {}
        call_prompt = call_completion = 0
        for span in trace["spans"]:
            if span["span_kind"] != "LLM":
                continue
            counts = attr(span, "llm", "token_count") or {}
            call_prompt += int(counts.get("prompt") or 0)
            call_completion += int(counts.get("completion") or 0)
        reconciliation.append({
            "trace_id": trace["trace_id"],
            "turn_prompt": turn.get("prompt"),
            "llm_span_prompt_sum": call_prompt,
            "delta_prompt": (turn.get("prompt") or 0) - call_prompt,
            "turn_completion": turn.get("completion"),
            "llm_span_completion_sum": call_completion,
            "delta_completion": (turn.get("completion") or 0) - call_completion,
        })

    parenting = []
    for trace in traces:
        by_id = {s["span_id"]: s for s in trace["spans"]}
        misparented = []
        for span in trace["spans"]:
            parent = by_id.get(span["parent_id"]) if span["parent_id"] else None
            parent_name = parent["name"] if parent else "(root)"
            if span["span_kind"] == "LLM" and not parent_name.startswith("iteration."):
                misparented.append(f"{span['name']} under {parent_name}")
            if span["span_kind"] == "TOOL" and not parent_name.startswith("iteration."):
                misparented.append(f"{span['name']} under {parent_name}")
        parenting.append({"trace_id": trace["trace_id"],
                          "spans_not_under_an_iteration": misparented})

    ttfts = [t for trace in traces
             for t in _all_ttft(trace)]
    return {
        "corpus": {
            "traces": len(traces),
            "spans": sum(t["span_count"] for t in traces),
            "window_utc": [traces[0]["started_utc"], traces[-1]["started_utc"]],
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
        "ttft_ms": {"n": len(ttfts),
                    "min": min(ttfts) if ttfts else None,
                    "median": int(statistics.median(ttfts)) if ttfts else None,
                    "max": max(ttfts) if ttfts else None},
    }


#: Bash commands that read the filesystem. The tool policy denies `Read`,
#: `Glob` and `Grep` outright — `sim/agent/claude_code.py` says why: the corpus
#: is a directory of files and one of them is the answer key. Whether the `Bash`
#: form is equally denied is a question about the CLI's matcher, not about our
#: allowlist, so it is asked of the data rather than assumed.
FILE_READING_COMMANDS = ("grep", "cat", "head", "tail", "less", "awk", "sed",
                         "find", "ls ", "od ", "xxd", "strings")


def _bash_file_reads(traces: list[dict]) -> list[dict[str, Any]]:
    found = []
    for trace in traces:
        for span in trace["spans"]:
            if span["name"] != "tool.Bash":
                continue
            raw = attr(span, "input", "value") or ""
            try:
                command = json.loads(raw).get("command", "")
            except (ValueError, TypeError, AttributeError):
                command = str(raw)
            output = attr(span, "output", "value") or ""
            denied = "Permission to use Bash has been denied" in output
            first = command.strip().split()[0] if command.strip() else ""
            if any(command.strip().startswith(c) for c in FILE_READING_COMMANDS):
                found.append({
                    "trace_id": trace["trace_id"],
                    "command_head": command[:200],
                    "binary": first,
                    "denied": denied,
                    "output_bytes": len(output),
                })
    return found


def findings(traces: list[dict], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    layers = analysis["layer_totals"]
    if layers["iteration_CHAIN"] == layers["LLM"] and layers["LLM"]:
        out.append({
            "id": "iteration-level-exists",
            "kind": "verification",
            "statement": f"Every model call sits under its own iteration span: "
                         f"{layers['iteration_CHAIN']} iteration.N spans and "
                         f"{layers['LLM']} LLM spans across "
                         f"{analysis['corpus']['traces']} traces.",
            "evidence": "_analysis.layer_totals",
            "reading": "docs/span-schema.md called this level unobservable "
                       "outside the CLI. It is observable: the stream names the "
                       "message, and one message id is one iteration.",
        })
    bad = [p for p in analysis["parenting"] if p["spans_not_under_an_iteration"]]
    out.append({
        "id": "tree-is-fully-parented",
        "kind": "verification",
        "statement": f"{len(traces) - len(bad)}/{len(traces)} traces have every "
                     f"LLM and TOOL span filed under an iteration.",
        "evidence": "_analysis.parenting",
        "reading": "Correlation is by message id, which the stream and the "
                   "transcript both name — exact, not a time heuristic.",
    })
    off = [r for r in analysis["token_reconciliation"]
           if r["delta_prompt"] or r["delta_completion"]]
    out.append({
        "id": "llm-spans-reconcile-exactly",
        "kind": "verification",
        "statement": f"Per-call tokens sum exactly to the turn totals on "
                     f"{len(traces) - len(off)}/{len(traces)} traces.",
        "evidence": "_analysis.token_reconciliation",
        "reading": "The transcript is read once, not twice, and no call is "
                   "dropped.",
    })
    reasoning = analysis["reasoning"]
    out.append({
        "id": "reasoning-is-present",
        "kind": "verification",
        "statement": f"{reasoning['spans_with_reasoning']}/"
                     f"{reasoning['llm_spans']} LLM spans carry non-empty "
                     f"reasoning ({reasoning['reasoning_chars']} chars).",
        "evidence": "_analysis.reasoning, traces[].reasoning.reasoning_texts",
        "reading": "Headless sessions persist reasoning in full. Before "
                   "2026-08-09 all of this was parsed and discarded.",
    })
    reads = _bash_file_reads(traces)
    allowed = [r for r in reads if not r["denied"]]
    if allowed:
        out.append({
            "id": "bash-file-read-bypasses-the-tool-policy",
            "kind": "defect",
            "statement": f"{len(allowed)} Bash calls read files and were NOT "
                         f"refused, using: "
                         f"{sorted({r['binary'] for r in allowed})}. The tool "
                         f"policy denies Read/Glob/Grep precisely so the agent "
                         f"has no file reader.",
            "evidence": "_findings_detail.bash_file_reads",
            "reading": "The matcher permits some read-only Bash commands "
                       "regardless of --allowed-tools. The module docstring "
                       "records probing echo/pwd/ls/whoami; these were not "
                       "probed. The corpus is not mounted into the agent, so "
                       "the answer key stays out of reach — but the agent can "
                       "re-read its own session transcript and any tool result "
                       "the CLI spilled to disk, which is a way round its own "
                       "context window and a confound for any question about "
                       "context strategy or token cost.",
        })
    return out


def _all_ttft(trace: dict) -> list[int]:
    out = []
    for span in trace["spans"]:
        value = attr(span, "b2e", "llm", "ttft_ms")
        if value is not None:
            out.append(int(value))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since-rowid", type=int, required=True)
    parser.add_argument("--project", default="b2e-sim")
    parser.add_argument("--exported-at", required=True,
                        help="UTC stamp, e.g. 2026-08-09T06:00:00Z. Passed in "
                             "rather than read from the clock so the file is "
                             "reproducible.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    built = build(args.since_rowid, args.project)
    if not built:
        print("no traces in that range", file=sys.stderr)
        return 1
    traces = built["traces"]
    analysis = analyse(traces)

    document = {
        "_about": {
            "source": f"Arize Phoenix (arizephoenix/phoenix:version-19.13.0-"
                      f"nonroot), project '{args.project}', read from backing "
                      f"Postgres {CONTAINER}",
            "exported_at_utc": args.exported_at,
            "selection": f"every trace with phoenix rowid >= {args.since_rowid}. "
                         f"Nothing is sampled or truncated: all spans, all "
                         f"attributes, full payloads.",
            "format": "the span tree in docs/span-schema.md for "
                      "harness=claude_code, as of the reasoning-tracing change: "
                      "AGENT root, CHAIN iteration.N per model call, LLM under "
                      "its iteration carrying reasoning and a measured ttft, "
                      "TOOL under the iteration that asked for it, heimdall.* "
                      "and sandbox.execute under their tool call.",
            "caveats": [
                "llm.input_messages is a RECONSTRUCTION, not the request — see "
                "b2e.llm.prompt_reconstruction on every LLM span.",
                "b2e.llm.timing='derived': the LLM span window comes from two "
                "transcript timestamps, nothing timed the request. "
                "b2e.llm.ttft_ms is the one measured latency.",
                "b2e.turn.api_duration_ms is the CLI's own figure and is not a "
                "wall-clock slice of the turn.",
            ],
        },
        "_analysis": analysis,
        "_findings": findings(traces, analysis),
        "_findings_detail": {"bash_file_reads": _bash_file_reads(traces)},
        "traces": traces,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(traces)} traces, "
          f"{document['_analysis']['corpus']['spans']} spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
