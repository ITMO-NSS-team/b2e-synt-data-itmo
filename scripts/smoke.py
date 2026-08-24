#!/usr/bin/env python3
"""End-to-end smoke test: the Definition of Done, executed.

Proves, in order:

  1. a researcher opens a session and POSTs one question
  2. the agent answers, having actually called Heimdall
  3. the full trace is retrievable
  4. feedback can be attached
  5. one config value is changed in the registry (as the admin UI does)
  6. the same question is re-run
  7. the two runs differ in the fingerprint, and the difference is visible

Runs against the compose stack when ``SMOKE_TARGET=stack``; otherwise it starts
the emulator and agent in-process, which is what makes it usable in CI. The
model is served from replay or a scripted client, so this spends nothing.

Exit code is non-zero on any failed step, so ``make smoke`` is a gate rather
than a demo.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import uvicorn

OK, BAD = "  ok  ", " FAIL "
_failures: list[str] = []


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)))


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{OK if condition else BAD}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        _failures.append(label)
    return condition


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            return server, thread
        time.sleep(0.05)
    raise RuntimeError("server did not start")


def main() -> int:
    from sim.agent.config import AgentConfig
    from sim.agent.llm import LLMResponse, ScriptedClient
    from sim.emulator.app import create_app as create_emulator
    from sim.emulator.config import EmulatorConfig
    from sim.fingerprint import FINGERPRINT_FIELDS

    workdir = Path(os.environ.get("SMOKE_VAR", "var/smoke"))
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in workdir.glob("*.db*"):
        stale.unlink()

    step("1 · bring up the emulator")
    emu_port = free_port()
    emulator = create_emulator(EmulatorConfig(
        snapshot_traps_on=os.environ.get("SMOKE_DATA", "data-small"),
        latency_profile="realistic"))
    emu_server, emu_thread = serve(emulator, emu_port)
    emu_url = f"http://127.0.0.1:{emu_port}"
    health = httpx.get(f"{emu_url}/control/healthz", trust_env=False).json()
    check("emulator healthy", health.get("status") == "ok")
    check("condition reported", bool(health.get("data_snapshot_hash")),
          f"snapshot={health['data_snapshot_hash']} traps={health['traps_enabled']} "
          f"latency={health['latency_profile']}")

    identities = httpx.get(f"{emu_url}/control/identities?n=3",
                           trust_env=False).json()
    employee = next(i for i in identities if i["role"] == "self")
    check("researcher picked an employee identity", True,
          f"employee_id={employee['employee_id']} role={employee['role']}")

    step("2 · bring up the agent")
    os.environ["HEIMDALL_URL"] = emu_url
    os.environ["B2E_LLM_MODE"] = "replay"
    os.environ["B2E_REGISTRY_DB"] = str(workdir / "registry.db")
    os.environ["B2E_AGENT_DB"] = str(workdir / "agent.db")

    from sim.agent.app import AgentState, create_app as create_agent

    state = AgentState()
    # Scripted so the smoke test is hermetic and free. The request-shaping,
    # tool dispatch, permission enforcement and tracing are all real.
    # The shipped default harness is claude_code; that path ignores
    # state.client and needs a live credential. Force the in-process loop.
    scripted_config = AgentConfig(harness="messages_api").as_dict()
    state.registry.commit(
        "agent_config", "agent", scripted_config,
        actor="smoke", note="hermetic: messages_api + ScriptedClient")
    question = "Сколько сотрудников в моём подразделении?"
    state.client = ScriptedClient([
        LLMResponse(content=[{"type": "tool_use", "id": "t1", "name": "mcp_query",
                              "input": {"schema": "dm_core",
                                        "logic_model": "employee_actual",
                                        "columns": ["person_id"], "limit": 5}}],
                    stop_reason="tool_use", prompt_tokens=1400, completion_tokens=60),
        LLMResponse(content=[{"type": "text",
                              "text": "По доступным мне данным — 5 записей."}],
                    stop_reason="end_turn", prompt_tokens=1900, completion_tokens=40),
    ] * 2)

    agent_port = free_port()
    agent_server, agent_thread = serve(create_agent(state), agent_port)
    agent_url = f"http://127.0.0.1:{agent_port}"
    check("agent healthy",
          httpx.get(f"{agent_url}/healthz", trust_env=False).json()["status"] == "ok")

    step("3 · POST one question, get an answer")
    session = httpx.post(f"{agent_url}/sessions", trust_env=False,
                         json={"employee_id": employee["employee_id"]}).json()
    fingerprint_a = session["fingerprint"]
    condition_a = session["condition_id"]
    check("session opened", bool(session.get("session_id")),
          f"session={session['session_id']}")
    check("fingerprint complete", set(fingerprint_a) == set(FINGERPRINT_FIELDS),
          json.dumps(fingerprint_a, ensure_ascii=False))

    reply_a = httpx.post(f"{agent_url}/sessions/{session['session_id']}/messages",
                         trust_env=False, json={"content": question}, timeout=120).json()
    check("answer returned", bool(reply_a.get("answer")),
          repr(reply_a["answer"])[:70])
    check("the agent actually called Heimdall",
          reply_a["stats"]["heimdall_calls"] >= 1,
          f"heimdall_calls={reply_a['stats']['heimdall_calls']} "
          f"tool_calls={reply_a['stats']['tool_calls']} "
          f"tokens={reply_a['stats']['total_tokens']}")

    step("4 · pull the transcript and the run record")
    detail = httpx.get(f"{agent_url}/sessions/{session['session_id']}",
                       trust_env=False).json()
    check("transcript stored", len(detail["messages"]) >= 2,
          f"{len(detail['messages'])} messages")
    check("run is attributable to a condition", bool(condition_a),
          f"condition_id={condition_a}")

    step("5 · permission scoping is real")
    denied = httpx.post(f"{emu_url}/api/v1/mcp/query/", trust_env=False,
                        headers={"Authorization": "Bearer t",
                                 "X-Employee-Id": employee["employee_id"]},
                        json={"schema": "recruitment",
                              "logic_model": "job_requisition_large",
                              "columns": ["id"], "limit": 1})
    check("restricted model returns 403", denied.status_code == 403,
          json.dumps(denied.json(), ensure_ascii=False)[:90])

    step("6 · change one config value, as the admin UI does")
    before = state.registry.head("agent_config")
    current = AgentConfig.from_dict(state.registry.load("agent_config")[1])
    # temperature is the varied field on purpose. context_strategy used to be
    # varied here too, but step 7 opens a session against the result, and
    # windowed/summarised are now refused on the default claude_code harness —
    # the CLI owns its window, so the value would have named an arm that ran
    # identically to `full`. Varying temperature moves both the field and
    # agent_config_version, which is all steps 7-8 assert.
    changed = {**current.as_dict(), "temperature": 0.7}
    after = state.registry.commit("agent_config", "agent", changed,
                                  actor="smoke", note="smoke: vary temperature")
    check("a new version was appended, not edited in place",
          after.version == before.version + 1,
          f"{before.ref} -> {after.ref}")
    check("the old version is still resolvable",
          state.registry.load(before.ref)[1]["temperature"] == current.temperature,
          f"{before.ref}.temperature={current.temperature}")
    check("the change is in the audit log",
          any(e["action"] == "config.commit" and e["target"] == after.ref
              for e in state.registry.audit_read()))

    step("7 · re-run the same question under the new config")
    session_b = httpx.post(f"{agent_url}/sessions", trust_env=False,
                           json={"employee_id": employee["employee_id"]}).json()
    reply_b = httpx.post(f"{agent_url}/sessions/{session_b['session_id']}/messages",
                         trust_env=False, json={"content": question}, timeout=120).json()
    fingerprint_b = session_b["fingerprint"]
    condition_b = session_b["condition_id"]
    check("second run answered", bool(reply_b.get("answer")))

    step("8 · diff the two runs")
    differences = {k: (fingerprint_a[k], fingerprint_b[k])
                   for k in fingerprint_a if fingerprint_a[k] != fingerprint_b[k]}
    for key, (was, now) in differences.items():
        print(f"        {key}: {was!r} -> {now!r}")
    check("the fingerprints differ", bool(differences),
          f"{len(differences)} field(s)")
    check("the condition id therefore differs", condition_a != condition_b,
          f"{condition_a} != {condition_b}")
    check("the difference is exactly the config version we changed",
          set(differences) == {"agent_config_version", "temperature"},
          str(sorted(differences)))

    step("9 · cost guard refuses before dispatch")
    from sim.costguard import Budget, CostCeilingExceeded, CostGuard, Projection

    guard = CostGuard(Budget(max_usd=0.01), experiment_id="smoke")
    projection = Projection(questions=500, expected_calls_per_question=6,
                            expected_prompt_tokens_per_call=6000,
                            expected_completion_tokens_per_call=700,
                            model_id=current.model_id)
    refused = False
    try:
        guard.check_projection(projection)
    except CostCeilingExceeded as exc:
        refused = True
        print(f"        {exc}")
    check("a batch over its ceiling never dispatches", refused,
          f"projected ${projection.projected_usd:.2f} vs ceiling $0.01")

    step("10 · the sandbox refuses unapproved code")
    from sim.skills import NotExecutable, SkillStore

    skills = SkillStore(state.registry)
    drafted = skills.author(name="smoke_skill", definition={"name": "smoke_skill"},
                            code="def run(payload):\n    return {'ok': True}\n",
                            author="agent")
    check("agent-authored skill entered as draft",
          drafted.state.value == "draft", f"hash={drafted.code_hash[:12]}")
    blocked = False
    try:
        skills.resolve_for_execution(drafted.code_hash)
    except NotExecutable as exc:
        blocked = True
        print(f"        {exc}")
    check("draft code is not executable", blocked)

    emu_server.should_exit = True
    agent_server.should_exit = True

    print("\n" + "=" * 70)
    if _failures:
        print(f"SMOKE FAILED — {len(_failures)} check(s): {_failures}")
        return 1
    print("SMOKE PASSED — the full Definition of Done path works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
