#!/usr/bin/env python3
"""Golden path against a running compose stack: question → answer → trace → feedback.

Picks a manager identity from data-small so "how many people in my unit?" is
in scope, opens a session on ``agent_config_openrouter``, and fails if the
model never called Heimdall — a guessed number is not a working MVP.

Requires ``deploy/.env`` with RESEARCHER_PASSWORD and a live OpenRouter key
already in the running agent. Spends free-tier tokens.

    make seed && make up && make demo
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from sim.agent.shipped import OPENROUTER_CONFIG_REF
from sim.emulator.identity import IdentityIndex

QUESTION = "Сколько сотрудников в моём подразделении?"
OK, BAD = "  ok  ", " FAIL "
_failures: list[str] = []


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)))


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{OK if condition else BAD}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        _failures.append(label)
    return condition


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def pick_manager(data_dir: Path) -> dict:
    index = IdentityIndex(data_dir)
    identities = index.sample_identities(5)
    manager = next((i for i in identities if i["role"] == "manager"), None)
    if manager is None:
        raise SystemExit("no manager identity in the corpus; rebuild with make seed")
    return manager


def proxy_client(base: str, *, public_host: str, auth: tuple[str, str],
                 timeout: float) -> httpx.Client:
    """Talk to the edge proxy without needing working container DNS.

    Caddy's site block is ``PUBLIC_HOST`` (localhost). Connecting to
    ``https://127.0.0.1`` with that Host empty-200s. Resolving ``localhost``
    fails in a non-root container on this host because Docker bind-mounts
    ``/etc/hosts`` from NFS. Connect by loopback IP, send the name Caddy
    expects, and stay on HTTP/1.1 so that Host header is not ignored.
    """
    parsed = urlparse(base)
    hostname = parsed.hostname or public_host or "localhost"
    header_host = public_host or hostname
    if hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = f"127.0.0.1:{parsed.port}" if parsed.port else "127.0.0.1"
        connect = urlunparse(parsed._replace(netloc=netloc))
    else:
        connect = base
    return httpx.Client(
        base_url=connect, auth=auth, verify=False, timeout=timeout,
        trust_env=False, http2=False, headers={"Host": header_host})


def main() -> int:
    env = {**load_dotenv(Path("deploy/.env")), **os.environ}
    base = env.get("PUBLIC_URL", "https://localhost:8443").rstrip("/")
    password = env.get("RESEARCHER_PASSWORD", "")
    config_ref = env.get("DEMO_CONFIG_REF", OPENROUTER_CONFIG_REF)
    timeout = float(env.get("DEMO_TIMEOUT", "180"))
    data_dir = Path(env.get("DEMO_DATA", "data-small"))

    if not password:
        print("RESEARCHER_PASSWORD is empty. Put the researcher plaintext in "
              "deploy/.env (the value that matches BASIC_AUTH_HASH).")
        return 2
    if not data_dir.joinpath("manifest.json").exists():
        print(f"no corpus at {data_dir}; run `make seed` first")
        return 2

    auth = ("researcher", password)
    client = proxy_client(
        base, public_host=env.get("PUBLIC_HOST", "localhost"),
        auth=auth, timeout=timeout)

    step("1 · pick a manager identity")
    manager = pick_manager(data_dir)
    check("manager identity", True,
          f"employee_id={manager['employee_id']} visible={manager['visible_people']}")

    step("2 · stack is reachable")
    try:
        health = client.get("/agent/healthz")
    except httpx.HTTPError as exc:
        check("agent healthz", False, str(exc))
        print("Is the stack up? `make up`, then retry.")
        return 1
    check("agent healthz", health.status_code == 200, health.text[:120])
    if health.status_code == 401:
        print("Basic auth rejected. RESEARCHER_PASSWORD must match BASIC_AUTH_HASH.")
        return 1

    step("3 · POST one question, get an answer")
    session = client.post("/agent/sessions", json={
        "employee_id": str(manager["employee_id"]),
        "config_ref": config_ref,
        "metadata": {"role": manager["role"]},
    })
    if session.status_code >= 400 or not session.content:
        check("session opened", False,
              session.text[:300] or f"empty {session.status_code} from {base} "
              "(Caddy matches PUBLIC_HOST=localhost; 127.0.0.1 is a different Host)")
        return 1
    body = session.json()
    session_id = body.get("session_id")
    check("session opened", bool(session_id),
          f"session={session_id} config={config_ref}")
    check("fingerprint names the OpenRouter config",
          (body.get("fingerprint") or {}).get("agent_config_version", "").startswith(
              OPENROUTER_CONFIG_REF) or config_ref != OPENROUTER_CONFIG_REF,
          json.dumps(body.get("fingerprint"), ensure_ascii=False)[:200])

    reply = client.post(f"/agent/sessions/{session_id}/messages",
                        json={"content": QUESTION})
    if reply.status_code >= 400:
        check("answer returned", False, reply.text[:400])
        return 1
    answer = reply.json()
    stats = answer.get("stats") or {}
    check("answer returned", bool(answer.get("answer")),
          repr(answer.get("answer", ""))[:120])
    check("the agent actually called Heimdall",
          int(stats.get("heimdall_calls") or 0) >= 1,
          f"heimdall_calls={stats.get('heimdall_calls')} "
          f"tool_calls={stats.get('tool_calls')} "
          f"tokens={stats.get('total_tokens')}")
    if int(stats.get("heimdall_calls") or 0) < 1:
        print("The model answered without tools. stealth/ox-alpha needs "
              "reasoning.effort=low (sent automatically) and enough "
              "max_output_tokens for a tool call after thinking. In /admin/config "
              "raise max_output_tokens (shipped default is 16384), save as a "
              "new version, and rerun with DEMO_CONFIG_REF=agent_config_openrouter@N.")

    step("4 · pull the trace")
    trace = None
    for attempt in range(12):
        response = client.get(f"/research/traces/{session_id}")
        if response.status_code == 200:
            trace = response.json()
            break
        time.sleep(1.0)
    check("trace stored in Phoenix", trace is not None,
          f"span_count={(trace or {}).get('span_count')}" if trace
          else f"last={response.status_code} {response.text[:120]}")

    step("5 · leave feedback")
    feedback = client.post("/research/feedback", json={
        "session_id": session_id,
        "label": "like" if int(stats.get("heimdall_calls") or 0) >= 1 else "dislike",
        "explanation": "golden-path demo",
        "annotator": "demo",
    })
    check("feedback attached", feedback.status_code in (200, 201),
          feedback.text[:160])

    print("\n" + "=" * 70)
    if _failures:
        print(f"DEMO FAILED — {len(_failures)} check(s): {_failures}")
        return 1
    print("DEMO PASSED — request → agent → Heimdall → answer → trace → feedback.")
    print(f"Phoenix: {base}/phoenix/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
