#!/usr/bin/env python3
"""Build a standalone HTML trace viewer from a Phoenix trace export.

The viewer is one self-contained file: the JSON is embedded, nothing is fetched
at runtime, so the output works offline and can be emailed or committed as-is.

    python3 build_viewer.py phoenix-traces-2026-08-08.json
    python3 build_viewer.py traces.json -o /tmp/report.html
    python3 build_viewer.py traces.json --open

All three export generations are understood:
  * the 2026-08-09 format — an iteration.N CHAIN between the turn and each model
    call, span durations in duration_ms, LLM spans carrying real content (system
    prompt, input messages, reasoning, the tool calls the model asked for) and a
    measured ttft, plus _findings_detail. The per-trace tool_call_counts and
    heimdall_* roll-ups are gone; the viewer recounts them from the spans;
  * the 2026-08-08 format — AGENT root, LLM spans, TOOL spans, heimdall.* CHAIN
    children, durations in duration_s, token counts, _analysis / _findings;
  * the older …-sample.json format — no LLM layer, tool durations recorded as
    0 s and every child span stamped with the turn's end time. The viewer
    detects that and shows an ordered call sequence instead of a fake timeline.

Anything shaped like {"traces": [...]} with OpenInference spans will render;
unknown top-level keys are ignored rather than fatal.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections import Counter
from pathlib import Path

#: The template moved under ``sim/`` so the Docker image carries it — the admin
#: UI serves the same page live, and ``deploy/Dockerfile`` copies ``sim`` but
#: not the repository root. The old location is still accepted so a copy of this
#: script sitting beside a template keeps working.
_HERE = Path(__file__).resolve().parent
TEMPLATE = next((p for p in (_HERE / "sim" / "viewer" / "template.html",
                             _HERE / "template.html") if p.is_file()),
                _HERE / "sim" / "viewer" / "template.html")
PLACEHOLDER = "__DATA__"


def load_export(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        sys.exit(f"error: no such file: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: {path} is not valid JSON — {exc}")

    if not isinstance(data, dict) or not isinstance(data.get("traces"), list):
        sys.exit(
            f"error: {path} does not look like a Phoenix export "
            '(expected a top-level object with a "traces" array)'
        )
    if not data["traces"]:
        sys.exit(f"error: {path} contains no traces")
    return data


def build(data: dict, template: str) -> str:
    # `</` is escaped so a payload containing "</script>" cannot close the
    # embedding tag; `<\/` is a legal JSON escape, so the data is unchanged.
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)


def describe(data: dict) -> str:
    traces = data["traces"]
    spans = [s for t in traces for s in (t.get("spans") or [])]
    bits = [f"{len(traces)} traces", f"{len(spans)} spans"]

    kinds = Counter(s.get("span_kind") for s in spans)
    layers = ", ".join(f"{n} {k}" for k, n in kinds.most_common() if k)
    if layers:
        bits.append(layers)

    # Reasoning is the headline of the 2026-08-09 generation — say whether it is there.
    chars = sum(
        len(c.get("message_content", {}).get("text") or "")
        for s in spans
        for m in ((s.get("attributes") or {}).get("llm") or {}).get("output_messages") or []
        for c in (m.get("message") or {}).get("contents") or []
        if c.get("message_content", {}).get("type") == "reasoning"
    )
    if chars:
        bits.append(f"{chars:,} chars of reasoning")

    if data.get("_findings"):
        bits.append(f"{len(data['_findings'])} findings")
    exported = (data.get("_about") or {}).get("exported_at_utc")
    if exported:
        bits.append(f"exported {exported}")
    return ", ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a standalone HTML viewer from a Phoenix trace export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The generated page also has a 'Load JSON…' button, so an existing\n"
               "viewer can open another export without rebuilding.",
    )
    ap.add_argument("traces", type=Path, help="Phoenix trace export (.json)")
    ap.add_argument("-o", "--output", type=Path,
                    help="output HTML path (default: <input>-viewer.html next to the input)")
    ap.add_argument("--open", dest="open_browser", action="store_true",
                    help="open the result in the default browser when done")
    args = ap.parse_args()

    if not TEMPLATE.is_file():
        sys.exit(f"error: template not found at {TEMPLATE} — keep it beside this script")
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        sys.exit(f"error: {TEMPLATE} has no {PLACEHOLDER} placeholder to fill")

    data = load_export(args.traces)
    out = args.output or args.traces.with_name(args.traces.stem + "-viewer.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(data, template), encoding="utf-8")

    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  {describe(data)}")
    if args.open_browser:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
