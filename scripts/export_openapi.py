#!/usr/bin/env python3
"""Export the OpenAPI specs for b2e-agent and research-api.

Generated from the running FastAPI applications rather than hand-written. A
hand-written spec is a second source of truth that drifts the first time someone
adds a field, and a contract nobody can trust is worse than no contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path("docs/openapi")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("B2E_LLM_MODE", "replay")
    os.environ.setdefault("B2E_REGISTRY_DB", "var/openapi-export.db")
    os.environ.setdefault("B2E_AGENT_DB", "var/openapi-export-agent.db")

    from sim.agent.app import AgentState, create_app as create_agent
    from sim.research.app import ResearchState, create_app as create_research

    written = []
    for name, factory, state in (
        ("b2e-agent", create_agent, AgentState()),
        ("research-api", create_research, ResearchState()),
    ):
        spec = factory(state).openapi()
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), "utf-8")
        paths = len(spec.get("paths", {}))
        written.append((name, path, paths))
        print(f"{name:14s} {paths:3d} paths -> {path}")

    if not written:
        raise SystemExit("nothing exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
