#!/usr/bin/env python3
"""Extract the recruitment RPC route table from the Heimdall OpenAPI spec.

The raw specification is an internal document and stays out of git (see
``.gitignore``). The derived route table does ship, on the same principle as
``catalog/snapshot.json``: without it the emulator cannot reproduce the real
surface, and it contains operation names and shapes rather than content.

Usage:
    python scripts/extract_recruitment_rpc.py <Heimdall_openapi.json> \
        --out catalog/recruitment_rpc.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TAG = "MCP Recruitment"


def _params(op: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for p in op.get("parameters", []) or []:
        out.append({
            "name": p.get("name"),
            "in": p.get("in"),
            "required": bool(p.get("required")),
            "type": (p.get("schema") or {}).get("type", "string"),
        })
    return out


def _body_ref(op: dict[str, Any]) -> str | None:
    body = op.get("requestBody") or {}
    schema = (((body.get("content") or {}).get("application/json") or {}).get("schema") or {})
    ref = schema.get("$ref")
    return ref.rsplit("/", 1)[-1] if ref else None


def _required_body_fields(spec: dict[str, Any], name: str | None) -> list[str]:
    if not name:
        return []
    schema = (spec.get("components", {}).get("schemas", {}) or {}).get(name) or {}
    return list(schema.get("required") or [])


def extract(spec: dict[str, Any]) -> dict[str, Any]:
    ops: list[dict[str, Any]] = []
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if not isinstance(op, dict) or TAG not in (op.get("tags") or []):
                continue
            body_ref = _body_ref(op)
            ops.append({
                "operation_id": op.get("operationId"),
                "method": method.upper(),
                "path": path,
                "summary": (op.get("summary") or "").strip(),
                "parameters": _params(op),
                "body_schema": body_ref,
                "body_required": _required_body_fields(spec, body_ref),
            })
    ops.sort(key=lambda o: (o["path"], o["method"]))
    return {
        "tag": TAG,
        "count": len(ops),
        "envelope": {
            "success": {"result": "<payload>", "error": None},
            "error": {"result": None,
                      "error": {"code": "int|str", "message": "str", "data": "any"}},
        },
        "operations": ops,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("openapi")
    ap.add_argument("--out", default="catalog/recruitment_rpc.json")
    args = ap.parse_args()

    spec = json.loads(Path(args.openapi).read_text("utf-8"))
    table = extract(spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1), "utf-8")
    print(f"{table['count']} operations -> {out}")


if __name__ == "__main__":
    main()
