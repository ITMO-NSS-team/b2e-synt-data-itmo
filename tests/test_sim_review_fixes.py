"""Regressions for defects the pr-review-toolkit audit found.

Every one of these shipped while 171 tests were green, and each was invisible
for the same reason: the suite asserted on an internal value rather than on the
behaviour a researcher would observe. These tests assert the observable thing.
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from sim.agent.claude_code import HEIMDALL_TOOLS, ClaudeCodeHarness, parse_stream
from sim.agent.config import AgentConfig
from sim.agent.tools import KNOWN_TOOLS, tool_schemas
from sim.emulator.app import create_app
from sim.emulator.config import EmulatorConfig
from sim.emulator.identity import IdentityIndex, restrict, scope_filter

pytestmark = pytest.mark.skipif(
    not Path("data-small/manifest.json").exists(),
    reason="needs the data-small corpus; run `make seed`")

BEARER = {"Authorization": "Bearer t"}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(EmulatorConfig(snapshot_traps_on="data-small",
                                                latency_profile="instant")))


@pytest.fixture(scope="module")
def actors(client):
    ids = client.get("/control/identities?n=3").json()
    return {"manager": next(i for i in ids if i["role"] == "manager"),
            "employee": next(i for i in ids if i["role"] == "self")}


# ------------------------------------------------- row-level scope enforcement


def test_an_unfiltered_query_returns_only_rows_the_identity_may_see(client, actors):
    """The hole: `enforce` refused a query that NAMED a forbidden person, but a
    query with no person filter returned the whole mart. Measured before the
    fix: an identity entitled to 2 records received 50 rows, 49 of them other
    people's."""
    emp = actors["employee"]
    h = {**BEARER, "X-Employee-Id": emp["employee_id"]}
    r = client.post("/api/v1/mcp/query/", headers=h,
                    json={"schema": "dm_core", "logic_model": "employee_actual",
                          "columns": ["person_id"], "limit": 50})
    assert r.status_code == 200
    rows = r.json()["data"]
    assert rows, "an employee must still be able to read their own record"
    foreign = [x for x in rows if x.get("person_id") != emp["person_id"]]
    assert foreign == [], f"{len(foreign)} rows belong to other people"


def test_a_manager_sees_their_subtree_and_no_more(client, actors):
    mgr = actors["manager"]
    h = {**BEARER, "X-Employee-Id": mgr["employee_id"]}
    r = client.post("/api/v1/mcp/query/", headers=h,
                    json={"schema": "dm_core", "logic_model": "employee_actual",
                          "columns": ["person_id"], "limit": 500})
    assert r.status_code == 200
    got = {x["person_id"] for x in r.json()["data"]}
    scope = IdentityIndex("data-small").scope_for(mgr["employee_id"])
    assert got, "a manager must see somebody"
    assert got <= set(scope.visible_uuids)
    assert len(got) > 1, "a manager seeing only themselves is the collapsed-scope bug"


def test_naming_a_forbidden_person_is_still_a_403_not_an_empty_result(client, actors):
    """Kept deliberately distinct from row filtering: an empty result would
    assert the person does not exist, which is a different and wrong fact."""
    h = {**BEARER, "X-Employee-Id": actors["employee"]["employee_id"]}
    r = client.post("/api/v1/mcp/query/", headers=h,
                    json={"schema": "dm_core", "logic_model": "employee_actual",
                          "columns": ["person_id"],
                          "filters": {"type": "condition", "column": "person_id",
                                      "operator": "=",
                                      "value": actors["manager"]["person_id"]}})
    assert r.status_code == 403


def test_scope_filter_uses_one_key_space_per_column():
    """A mixed set produced `filter-value-invalid` — person_id is a UUID and
    employee_id is numeric — turning every scoped read into a 400."""
    scope = IdentityIndex("data-small").scope_for(
        IdentityIndex("data-small").employee_id[0])
    for column in ("person_id", "employee_id"):
        values = scope_filter(scope, column)["value"]
        assert values, column
        if column == "person_id":
            assert all("-" in v for v in values)
        else:
            assert all(v.isdigit() for v in values)


def test_restrict_ands_onto_existing_filters_without_dropping_them():
    scope = IdentityIndex("data-small").scope_for(
        IdentityIndex("data-small").employee_id[0])
    original = {"type": "condition", "column": "grade_level",
                "operator": ">", "value": 10}
    out = restrict(scope, {"filters": original}, column="person_id")
    assert out["filters"]["type"] == "and"
    assert original in out["filters"]["conditions"]


# ----------------------------------------------------------- latency injection


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_the_latency_profile_actually_delays_the_client():
    """RQ2's independent variable was a no-op: the middleware slept AFTER the
    response was already flushed, so `degraded` measured 4ms against `instant`'s
    3ms while the fingerprint labelled the runs as different conditions.

    Asserted against a real socket, because an in-process TestClient would not
    have caught the original bug either.
    """
    observed = {}
    for profile in ("instant", "degraded"):
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(
            create_app(EmulatorConfig(snapshot_traps_on="data-small",
                                      latency_profile=profile)),
            host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        url = f"http://127.0.0.1:{port}"
        ids = httpx.get(f"{url}/control/identities?n=3", trust_env=False).json()
        emp = next(i for i in ids if i["role"] == "self")
        h = {**BEARER, "X-Employee-Id": emp["employee_id"]}
        body = {"schema": "dm_core", "logic_model": "employee_actual",
                "columns": ["*"], "limit": 20}
        with httpx.Client(trust_env=False, timeout=120) as c:
            c.post(f"{url}/api/v1/mcp/query/", headers=h, json=body)
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                resp = c.post(f"{url}/api/v1/mcp/query/", headers=h, json=body)
                samples.append((time.perf_counter() - t0) * 1000)
            assert "x-b2e-injected-latency-ms" in resp.headers
        observed[profile] = sorted(samples)[len(samples) // 2]
        server.should_exit = True
        time.sleep(0.3)

    assert observed["degraded"] > observed["instant"] * 10, observed


# ------------------------------------------------------------- tool parity


def test_both_harnesses_expose_the_same_tools_for_one_config():
    """Under one agent_config_version the harnesses granted different tools —
    get_docs only in messages_api, get_overview only in claude_code — so a
    harness comparison was partly a comparison of capability surfaces."""
    config = AgentConfig()
    harness = ClaudeCodeHarness(heimdall_url="x", heimdall_token="y")
    via_messages = {t["name"] for t in tool_schemas(config.tool_subset)}
    via_claude = {t.split("__")[-1] for t in harness.allowed_tools(config)
                  if t.startswith("mcp__")}
    assert via_messages == via_claude


def test_an_unknown_tool_name_is_refused_not_silently_dropped():
    with pytest.raises(ValueError, match="unknown tools"):
        AgentConfig(tool_subset=("mcp_qeury",))


def test_heimdall_tools_matches_the_canonical_set():
    assert set(HEIMDALL_TOOLS) == set(KNOWN_TOOLS)


def test_the_bridge_serves_every_tool_the_harness_grants():
    """The harness names tools the MCP bridge must actually implement; a name in
    one and not the other is a tool the agent can request and never reach."""
    import re
    src = Path("heimdall/bridge.py").read_text("utf-8")
    served = set(re.findall(r'"name": "(\w+)"', src))
    assert set(HEIMDALL_TOOLS) <= served, set(HEIMDALL_TOOLS) - served


def test_argv_passes_the_deny_list_to_the_cli():
    """A surviving mutation deleted --disallowed-tools from argv and all 156
    tests stayed green: denied_tools() was tested six ways as a pure function
    and never asserted to reach the process."""
    harness = ClaudeCodeHarness(heimdall_url="x", heimdall_token="y")
    for config in (AgentConfig(), AgentConfig(code_execution="allowed")):
        argv = harness.build_argv("q", config=config, system_suffix="s",
                                  mcp_config_path="/tmp/m.json")
        assert "--disallowed-tools" in argv
        passed = argv[argv.index("--disallowed-tools") + 1].split(",")
        assert passed == harness.denied_tools(config)


# --------------------------------------------------------------- tool output


def test_the_harness_records_what_tools_returned_not_only_what_was_asked():
    """Tool spans carried input only, so there was no record of what the API
    returned — and the fabrication metric compares the answer against exactly
    that. With no evidence, every answer scored clean."""
    import json
    stream = "\n".join(json.dumps(e) for e in [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "mcp__heimdall__mcp_query",
             "input": {"schema": "dm_core"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": '{"data":[{"person_id":"abc"}]}'}]}]}},
        {"type": "result", "is_error": False, "num_turns": 2, "result": "ok",
         "usage": {"input_tokens": 1, "output_tokens": 1}},
    ])
    out = parse_stream(stream)
    assert len(out.tool_calls) == 1
    assert "person_id" in (out.tool_calls[0]["output"] or "")
