#!/usr/bin/env python
"""Export how each arm's memory changed from epoch to epoch.

The registry is append-only, so every intermediate state of every memory is
still there — ``memory_A3_fleet@1`` through ``@8`` are all fetchable. The head
version alone answers "what did it end up believing"; the series answers "how
did it get there", which is the question a reflection experiment is actually
about. A lesson that appears in epoch 2, survives three epochs and is then
dropped tells you something a final snapshot cannot.

Two modes, because the registry lives inside a container and the rendering does
not:

    # inside the agent container, which has the registry mounted
    python scripts/rq4_export_memory.py --dump > artifacts.json

    # on the host, from that file
    python scripts/rq4_export_memory.py --render artifacts.json > dynamics.md

Items are matched between consecutive versions by ``id``, so "added" and
"removed" mean what they say rather than "the text changed slightly".
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

MEMORY_NAMES = ("memory_A2_i1", "memory_A2_i2", "memory_A2_i3", "memory_A3_fleet")
ARM_OF = {"memory_A2_i1": "A2", "memory_A2_i2": "A2", "memory_A2_i3": "A2",
          "memory_A3_fleet": "A3"}


def _items(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return list(pack.get("items") or [])


def _key(item: dict[str, Any]) -> str:
    """Identity of a lesson across versions.

    Prefer the stable id; fall back to the text so a pack written before ids
    were populated still diffs sensibly rather than reporting every item as
    both added and removed on every epoch.
    """
    return str(item.get("id") or item.get("lesson_id") or item.get("text", ""))[:80]


def dump(registry_db: str) -> dict[str, Any]:
    sys.path.insert(0, "/app")
    from sim.registry import Registry

    reg = Registry(registry_db, readonly=True)
    out: dict[str, Any] = {"memories": {}, "run_design": None}

    for name in MEMORY_NAMES:
        head = reg.head(name)
        if head is None:
            continue
        series = []
        prev: dict[str, dict[str, Any]] = {}
        for version in range(1, head.version + 1):
            got = reg.get_version(name, version)
            if got is None:
                continue
            _, pack = reg.load(f"{name}@{version}")
            cur = {_key(i): i for i in _items(pack)}
            series.append({
                "version": version,
                # A pack committed after epoch N is what the agent reads during
                # epoch N+1, so the epoch it was BUILT from is version-1.
                "built_after_epoch": version - 1,
                "active_in_epoch": version,
                "n_items": len(cur),
                "rendered_chars": len(pack.get("rendered") or ""),
                "added": [cur[k] for k in cur if k not in prev],
                "removed": [prev[k] for k in prev if k not in cur],
                "kept": [k for k in cur if k in prev],
                "items": list(cur.values()),
                "rendered": pack.get("rendered"),
                "note": got.note,
            })
            prev = cur
        out["memories"][name] = {"arm": ARM_OF[name], "head": head.version,
                                 "series": series}

    rd = reg.head("run_design")
    if rd is not None:
        _, body = reg.load(f"run_design@{rd.version}")
        out["run_design"] = {"ref": f"run_design@{rd.version}", "body": body}
    return out


def render(doc: dict[str, Any]) -> str:
    lines: list[str] = ["# RQ4 — memory dynamics, epoch by epoch", ""]
    lines.append("A1 is absent by design: it carries no memory, which is the "
                 "point of it. `built_after_epoch` is the epoch whose episodes "
                 "produced the pack; `active_in_epoch` is when the agent read "
                 "it.")
    lines.append("")

    for name, blob in doc["memories"].items():
        lines += [f"## {name}  (arm {blob['arm']})", "",
                  "| epoch built | active in | items | chars | added | removed | kept |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for s in blob["series"]:
            lines.append(
                f"| {s['built_after_epoch']} | {s['active_in_epoch']} | "
                f"{s['n_items']} | {s['rendered_chars']} | {len(s['added'])} | "
                f"{len(s['removed'])} | {len(s['kept'])} |")
        lines.append("")
        for s in blob["series"]:
            if not s["added"] and not s["removed"]:
                continue
            lines.append(f"**after epoch {s['built_after_epoch']}**")
            for it in s["added"]:
                lines.append(f"- `+` *{it.get('kind','?')}* — {it.get('text','')}")
            for it in s["removed"]:
                lines.append(f"- `−` *{it.get('kind','?')}* — {it.get('text','')}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", action="store_true",
                   help="read the registry and print the JSON export")
    p.add_argument("--registry-db", default="/app/registry/registry.db")
    p.add_argument("--render", metavar="JSON",
                   help="render a previously dumped JSON as markdown")
    a = p.parse_args()

    if a.render:
        with open(a.render, encoding="utf-8") as fh:
            print(render(json.load(fh)))
        return
    if a.dump:
        print(json.dumps(dump(a.registry_db), ensure_ascii=False))
        return
    p.error("pass --dump or --render")


if __name__ == "__main__":
    main()
