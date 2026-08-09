"""Export Phoenix traces to a self-contained JSON file, with analysis.

Reads the backing Postgres directly rather than Phoenix's REST API: the API
paginates and reshapes, and the point of an export is to be the same bytes the
store holds. Nothing is sampled and no payload is truncated — unlike the live
explorer tab, which drops the reconstructed prompts to stay openable.

    python3 scripts/export_traces.py --since-rowid 61 \\
        --exported-at 2026-08-09T12:00:00Z -o var/phoenix-traces.json

The derived fields, the analysis and the findings come from ``sim.traceview``,
which the admin UI's explorer also uses. That is deliberate: the two paths
differed once already, and a viewer that disagrees with the export it was built
from is worse than no viewer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import traceview  # noqa: E402

CONTAINER = os.environ.get("B2E_PG_CONTAINER", "b2e-sim-postgres-1")
DB_USER = os.environ.get("B2E_PG_USER", "phoenix")
DB_NAME = os.environ.get("B2E_PG_DB", "phoenix")


def query(sql: str) -> list[dict]:
    """One query, returned as rows of dicts. Postgres does the JSON."""
    wrapped = f"select coalesce(json_agg(t), '[]'::json)::text from ({sql}) t"
    out = subprocess.run(
        ["docker", "exec", CONTAINER, "psql", "-U", DB_USER, "-d", DB_NAME,
         "-t", "-A", "-c", wrapped],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip() or "[]")


def build(since_rowid: int, project: str) -> list[dict]:
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
        return []

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
        trace["span_cost_usd_sum"] = round(
            cost_by_trace.get(trace["phoenix_rowid"], 0.0), 6)
        # Postgres already stores attributes nested, so no unflattening here —
        # the REST path is the one that needs it.
        trace["spans"] = [{k: v for k, v in s.items() if k != "trace_rowid"}
                          for s in rows]
        traceview.enrich(trace)
    return traces


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

    traces = build(args.since_rowid, args.project)
    if not traces:
        print("no traces in that range", file=sys.stderr)
        return 1

    document = traceview.build_document(traces, {
        "source": f"Arize Phoenix (arizephoenix/phoenix:version-19.13.0-"
                  f"nonroot), project '{args.project}', read from backing "
                  f"Postgres {CONTAINER}",
        "exported_at_utc": args.exported_at,
        "selection": f"every trace with phoenix rowid >= {args.since_rowid}. "
                     f"Nothing is sampled or truncated: all spans, all "
                     f"attributes, full payloads.",
        "format": "the span tree in docs/span-schema.md for "
                  "harness=claude_code, as of the reasoning-tracing change: "
                  "AGENT root, CHAIN iteration.N per model call, LLM under its "
                  "iteration carrying reasoning and a measured ttft, TOOL under "
                  "the iteration that asked for it, heimdall.* and "
                  "sandbox.execute under their tool call.",
        "caveats": [
            "llm.input_messages is a RECONSTRUCTION, not the request — see "
            "b2e.llm.prompt_reconstruction on every LLM span.",
            "b2e.llm.timing='derived': the LLM span window comes from two "
            "transcript timestamps, nothing timed the request. "
            "b2e.llm.ttft_ms is the one measured latency.",
            "b2e.turn.api_duration_ms is the CLI's own figure and is not a "
            "wall-clock slice of the turn.",
        ],
    })
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=1, default=str)
    print(f"wrote {args.out}: {len(traces)} traces, "
          f"{document['_analysis']['corpus']['spans']} spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
