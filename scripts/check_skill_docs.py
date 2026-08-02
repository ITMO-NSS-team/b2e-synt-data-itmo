"""Execute every JSON example in the skill library against a live emulator.

The reason this exists is the same one the harness note has a test: a document
that tells the agent something the API does not do is not merely unhelpful, it
walks the agent into a failure it cannot diagnose. The skill library is the
agent's map of the query contract, and an unexecuted example is an unverified
claim.

So every fenced ``json`` block that names a schema and a logic_model is POSTed
to ``mcp_query`` and expected to come back 200. Fragments that are not whole
request bodies — filter subtrees, response samples — are skipped by that same
test rather than by a hand-maintained exclusion list.

An example that genuinely cannot be executed as written — the vector-search body
needs 384 literal floats, which is not a thing to put in a document — is fenced
as ``jsonc`` instead, and the doc says in words that it illustrates a shape. The
fence language is the convention: ``json`` means "this runs", ``jsonc`` means
"this is a diagram". An exclusion list here would rot the first time a file was
renamed, and would hide the exclusion from the person reading the document.

Usage:
    python scripts/check_skill_docs.py [--url http://127.0.0.1:8081]
                                       [--employee 9877478]
Exit code is non-zero if any example fails, so it can gate a deploy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: The language tag must be exactly ``json`` — anchored to the end of the line
#: so that ``jsonc`` is not swallowed by a prefix match, which is precisely what
#: happened the first time and turned the illustrative block back into a
#: "malformed JSON" failure.
FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)```", re.S)


def examples(root: Path):
    """(file, label, body) for everything in the library that claims to run.

    Two sources, because the library has two shapes. A reference (``.md``)
    teaches a mechanism and carries fenced examples; a recipe (``.yaml``) *is* a
    query, and its ``query`` block is handed to the agent as "executes as is" —
    which is a promise worth checking, since the agent will not second-guess it.
    """
    for path in sorted(root.rglob("*.md")):
        for i, block in enumerate(FENCE.findall(path.read_text("utf-8"))):
            try:
                body = json.loads(block)
            except ValueError:
                yield path, f"#{i}", None    # malformed JSON is itself a defect
                continue
            if isinstance(body, dict) and body.get("schema") and body.get("logic_model"):
                yield path, f"#{i}", body

    import yaml

    for path in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))):
        try:
            meta = yaml.safe_load(path.read_text("utf-8")) or {}
        except Exception:                                 # noqa: BLE001
            yield path, ":yaml", None
            continue
        query = meta.get("query")
        if isinstance(query, dict) and query.get("schema") and query.get("logic_model"):
            yield path, ":query", query


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--employee", default="9877478")
    ap.add_argument("--token", default="sim-technical-account")
    ap.add_argument("--skills", default="heimdall-skills")
    args = ap.parse_args()

    import httpx

    headers = {"Authorization": f"Bearer {args.token}",
               "X-Employee-Id": args.employee,
               "Content-Type": "application/json"}
    client = httpx.Client(base_url=args.url, trust_env=False, timeout=180)

    checked = failed = 0
    for path, label, body in examples(Path(args.skills)):
        name = f"{path.name}{label}"
        if body is None:
            print(f"[FAIL] {name}: not parseable")
            failed += 1
            continue
        checked += 1
        response = client.post("/api/v1/mcp/query/", headers=headers, json=body)
        if response.status_code == 200:
            payload = response.json()
            rows = len(payload.get("data") or [])
            print(f"[ ok ] {name}: {rows} rows, has_next_page="
                  f"{payload.get('has_next_page')}")
        else:
            failed += 1
            detail = json.dumps(response.json(), ensure_ascii=False)[:220]
            print(f"[FAIL] {name}: {response.status_code} {detail}")

    print(f"\n{checked} executable examples, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
