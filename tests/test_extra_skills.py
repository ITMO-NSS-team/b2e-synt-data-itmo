"""Session catalog isolation through real HTTP boundaries, without a model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import socket
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
import uvicorn

from heimdall.app import (_State, _install_error_handler, _mcp_v1_router,
                          _mcp_v2_router)
from heimdall.skills.contexts import catalog_router
from sim.agent.app import create_app
from sim.agent.catalogs import SessionCatalogs
from sim.agent.claude_code import ClaudeCodeHarness
from sim.agent.config import AgentConfig
from sim.agent.loop import TurnResult
from sim.agent.opencode import OpenCodeHarness
from sim.agent.store import Store


def native(name: str, marker: str) -> dict:
    return {"filename": f"{name}.md", "content": (
        f"---\nname: {name}\ntitle: {marker}\nkind: reference\ndomain: general\n"
        f"description: {marker}\n---\nПрименяй {marker}.\n")}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("HEIMDALL_CATALOG_ADMIN_KEY", "private-service-key")
    base = native("existing", "baseline")
    root = tmp_path / "catalog"
    root.mkdir()
    (root / base["filename"]).write_text(base["content"])
    state = _State(None, None, root, None)
    app = FastAPI()
    _install_error_handler(app)
    app.include_router(_mcp_v1_router(state))
    app.include_router(_mcp_v2_router(state))
    app.include_router(catalog_router(state))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                          log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.01)
    assert server.started
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False)
    yield SimpleNamespace(state=state, app=app, client=client, url=str(client.base_url),
                          baseline=base, root=root)
    client.close()
    server.should_exit = True
    thread.join(timeout=5)
    assert not thread.is_alive()


def provision(catalog, files, employee="42"):
    return catalog.client.post("/internal/skill-catalogs", json={
        "employee_id": employee, "extra_skills": files},
        headers={"X-Catalog-Admin-Key": "private-service-key"})


def headers(context=None, employee="42"):
    result = {"Authorization": "Bearer t", "X-Employee-Id": employee}
    if context is not None:
        result["X-Skill-Catalog-Context"] = context
    return result


READS = ["/api/v2/mcp/overview/", "/api/v2/mcp/skills/?query=alpha",
         "/api/v2/mcp/skills/alpha/", "/api/v1/mcp/docs/?topic=alpha"]


def test_two_multiple_skill_catalogs_use_native_read_routes_and_one_index(catalog):
    c = catalog
    before = c.client.get(READS[0], headers=headers()).json()
    a = provision(c, [native("alpha", "alpha_unique"), native("beta", "beta_unique")])
    b = provision(c, [native("alpha", "other_unique"), native("gamma", "gamma_unique")])
    assert a.status_code == b.status_code == 201
    ca, cb = a.json()["catalog_context"], b.json()["catalog_context"]
    evidence = a.json()["catalog_evidence"]
    assert evidence["baseline_hash"] != evidence["catalog_hash"]
    assert evidence["native_skills"] == [
        {"filename": f"{name}.md", "name": name,
         "sha256": sha256(native(name, marker)["content"].encode()).hexdigest()}
        for name, marker in [("alpha", "alpha_unique"), ("beta", "beta_unique")]]
    detail = c.client.get(READS[2], headers=headers(ca)).json()
    assert detail["body"] == "Применяй alpha_unique.\n"
    assert c.client.get(READS[3], headers=headers(ca)).json()["markdown"] == detail["markdown"]
    assert c.client.get("/api/v1/mcp/docs/", headers=headers(ca)).json()["markdown"] == (
        "Доступные темы: alpha, beta, existing")
    assert c.client.get("/api/v2/mcp/skills/?query=alpha_unique", headers=headers(ca)).json() == (
        c.state.catalog_contexts.resolve(ca, "42").registry.find(query="alpha_unique"))
    assert c.client.get("/api/v2/mcp/skills/beta/", headers=headers(cb)).status_code == 404
    assert c.client.get(READS[2], headers=headers()).status_code == 404
    assert c.client.get(READS[0], headers=headers()).json() == before
    assert (c.root / "existing.md").read_text() == c.baseline["content"]
    assert list(c.root.iterdir()) == [c.root / "existing.md"]

    def read(i):
        context, marker = (ca, "alpha_unique") if i % 2 == 0 else (cb, "other_unique")
        answer = c.client.get(READS[2], headers=headers(context))
        assert answer.status_code == 200
        assert answer.json()["body"] == f"Применяй {marker}.\n"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(read, range(40)))


@pytest.mark.parametrize("files", [
    [native("existing", "collision")],
    [native("alpha", "one"), native("alpha", "two")],
    [native("alpha", "valid"), {"filename": "bad.md", "content": "---\nname: bad\n---\ninvalid"}],
    [{"filename": "../escape.md", "content": native("alpha", "valid")["content"]}],
    [{"filename": "recipe.yaml", "content": "name: bad"}],
    [{"filename": "bad.md", "content": "---\ninvalid: [\n---"}],
])
def test_invalid_batch_is_atomic_and_never_changes_baseline(catalog, files):
    assert provision(catalog, files).status_code == 422
    assert catalog.state.catalog_contexts._views == {}
    assert catalog.state.registry.all_names() == ["existing"]


def test_context_authentication_employee_binding_close_and_expiration(catalog, monkeypatch):
    c = catalog
    request = {"employee_id": "42", "extra_skills": [native("alpha", "a")]}
    assert c.client.post("/internal/skill-catalogs", json=request,
                         headers={"Authorization": "Bearer any"}).status_code == 403
    created = provision(c, request["extra_skills"]).json()
    context = created["catalog_context"]
    for path in READS:
        assert c.client.get(path, headers=headers(context, "other")).status_code == 403
        assert c.client.get(path, headers=headers("missing")).status_code == 410
        assert c.client.get(path, headers=headers("")).status_code == 410
    assert c.client.delete("/internal/skill-catalogs/current",
                           headers={"X-Skill-Catalog-Context": context}).status_code == 403
    assert c.client.delete("/internal/skill-catalogs/current", headers={
        "X-Skill-Catalog-Context": context,
        "X-Catalog-Admin-Key": "private-service-key"}).status_code == 204
    for path in READS:
        assert c.client.get(path, headers=headers(context)).status_code == 410
    context = provision(c, request["extra_skills"]).json()["catalog_context"]
    c.state.catalog_contexts._views[context].expires_at = time.monotonic() - 1
    assert c.client.get(READS[0], headers=headers(context)).status_code == 410
    assert c.state.catalog_contexts._views == {}


def test_session_followups_evidence_disposal_and_restart_fail_closed(catalog, tmp_path,
                                                                    fingerprint, monkeypatch):
    config = AgentConfig(harness="messages_api", conversation_mode="resume")
    state = SimpleNamespace(
        store=Store(tmp_path / "agent.db"), heimdall_url=catalog.url,
        heimdall_token="t", client=None,
        catalogs=SessionCatalogs(catalog.url, "private-service-key"),
        build_fingerprint=lambda ref: (replace(fingerprint, harness="messages_api"),
                                      config, "Employee {employee_id}"))
    calls = []
    def run_turn(**kwargs):
        tools = kwargs["tools"]
        calls.append(tools._client.headers.get("X-Skill-Catalog-Context"))
        detail = tools.dispatch("get_skill", {"name": "alpha"})
        return TurnResult(answer=detail["body"], stop_reason="end_turn",
                          iterations=1, tool_calls=1, heimdall_calls=1,
                          prompt_tokens=0, completion_tokens=0, cost_usd=0.0)
    monkeypatch.setattr("sim.agent.app.run_turn", run_turn)
    client = TestClient(create_app(state))
    first = client.post("/sessions", json={"employee_id": "42", "extra_skills": [
        native("alpha", "first"), native("beta", "first_b")]})
    assert first.status_code == 201
    sid = first.json()["session_id"]
    second = client.post("/sessions", json={"employee_id": "42", "extra_skills": [
        native("alpha", "second")]})
    sid2 = second.json()["session_id"]
    for content in ("first turn", "follow-up"):
        response = client.post(f"/sessions/{sid}/messages", json={"content": content})
        assert response.status_code == 200, response.text
        assert response.json()["answer"] == "Применяй first.\n"
        assert response.json()["catalog_evidence"] == first.json()["catalog_evidence"]
    assert calls[0] == calls[1]
    assert client.post(f"/sessions/{sid2}/messages", json={"content": "other"}).json()["answer"] == (
        "Применяй second.\n")
    assert calls[2] != calls[1]
    for response in (first, client.get("/sessions"), client.get(f"/sessions/{sid}")):
        assert all(context not in response.text for context in calls)
    with state.catalogs.turn(state.store.get_session(sid)):
        assert client.delete(f"/sessions/{sid}").status_code == 409
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.post(f"/sessions/{sid}/messages", json={"content": "after close"}).status_code == 410
    assert len(state.store.messages(sid)) == 4
    state.catalogs = SessionCatalogs(catalog.url, "private-service-key")
    assert client.post(f"/sessions/{sid2}/messages", json={"content": "after restart"}).status_code == 410
    assert len(state.store.messages(sid2)) == 2
    state.store.close()


def test_cli_bridge_environment_has_scoped_capability_and_no_admin_key(monkeypatch):
    monkeypatch.setenv("HEIMDALL_CATALOG_ADMIN_KEY", "private-service-key")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-model-key")
    config = AgentConfig(harness="claude_code")
    claude = ClaudeCodeHarness(heimdall_url="http://heimdall", heimdall_token="t")
    env = claude.mcp_config("42", config, catalog_context="capability")["mcpServers"]["heimdall"]["env"]
    assert env["HEIMDALL_CATALOG_CONTEXT"] == "capability"
    assert env["HEIMDALL_EMPLOYEE_ID"] == "42"
    assert "HEIMDALL_CATALOG_ADMIN_KEY" not in env
    assert "private-service-key" not in json.dumps(claude.child_env())
    opencode = OpenCodeHarness(heimdall_url="http://heimdall", heimdall_token="t")
    env = opencode._bridge_environment("42", config, "trace", "capability")
    assert env["HEIMDALL_CATALOG_CONTEXT"] == "capability"
    assert "HEIMDALL_CATALOG_ADMIN_KEY" not in env

    assert "private-service-key" not in json.dumps(opencode.child_env(Path("/tmp/test-config.json")))


def test_empty_additions_do_not_require_provisioning_key(tmp_path, fingerprint):
    state = SimpleNamespace(
        store=Store(tmp_path / "agent.db"),
        build_fingerprint=lambda ref: (fingerprint, None, "system"))
    client = TestClient(create_app(state))
    created = client.post("/sessions", json={"employee_id": "42", "extra_skills": []})
    assert created.status_code == 201
    assert "catalog_evidence" not in created.json()
    assert "skill_catalog" not in state.store.get_session(created.json()["session_id"])["fingerprint"]
    state.store.close()


def test_close_failure_is_retryable_and_closing_blocks_new_turns(monkeypatch):
    from fastapi import HTTPException
    catalogs = SessionCatalogs("http://unused", "service-key")
    catalogs.bind("s", "capability")
    session = {"id": "s", "employee_id": "42", "fingerprint": {"skill_catalog": {}}}
    entered, release = threading.Event(), threading.Event()
    calls = []

    def close(context):
        calls.append(context)
        if len(calls) == 1:
            raise HTTPException(503, "temporary outage")
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(catalogs, "close_context", close)
    with pytest.raises(HTTPException) as error:
        catalogs.close("s")
    assert error.value.status_code == 503
    assert catalogs._contexts["s"].capability == "capability"
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(catalogs.close, "s")
        assert entered.wait(timeout=5)
        with pytest.raises(HTTPException) as error:
            with catalogs.turn(session):
                pytest.fail("closing catalog allowed a new turn")
        assert error.value.status_code == 409
        release.set()
        closing.result(timeout=5)
    assert catalogs._contexts == {}
    assert catalogs._closing == set()
    assert calls == ["capability", "capability"]


def test_bridge_headers_preserve_explicit_catalog_context(monkeypatch):
    from heimdall import bridge
    monkeypatch.setattr(bridge, "CATALOG_CONTEXT", "private-capability")
    monkeypatch.setattr(bridge, "EMPLOYEE_ID", "42")
    assert bridge._headers()["X-Skill-Catalog-Context"] == "private-capability"
    assert bridge._headers()["X-Employee-Id"] == "42"
    monkeypatch.setattr(bridge, "CATALOG_CONTEXT", "")
    assert bridge._headers()["X-Skill-Catalog-Context"] == ""
    monkeypatch.setattr(bridge, "CATALOG_CONTEXT", None)
    assert "X-Skill-Catalog-Context" not in bridge._headers()


def test_idle_local_bindings_are_reaped_without_affecting_active_turns(monkeypatch):
    from fastapi import HTTPException
    catalogs = SessionCatalogs("http://unused", "key")
    catalogs.bind("expired", "old")
    catalogs.bind("busy", "active")
    catalogs._contexts["expired"].expires_at = time.monotonic() - 1
    catalogs._contexts["busy"].expires_at = time.monotonic() - 1
    catalogs._active.add("busy")
    catalogs.bind("new", "fresh")
    assert set(catalogs._contexts) == {"busy", "new"}
    with pytest.raises(HTTPException) as error:
        with catalogs.turn({"id": "expired", "employee_id": "42",
                            "fingerprint": {"skill_catalog": {}}}):
            pytest.fail("expired local binding resumed")
    assert error.value.status_code == 410


@pytest.mark.parametrize("harness_type", [ClaudeCodeHarness, OpenCodeHarness])
def test_real_cli_run_writes_scope_to_its_native_config(catalog, tmp_path, monkeypatch,
                                                      harness_type):
    from sim.agent.claude_code import ClaudeCodeResult
    created = provision(catalog, [native("alpha", "scoped")]).json()
    context = created["catalog_context"]
    harness_name = "claude_code" if harness_type is ClaudeCodeHarness else "open_code"
    config = AgentConfig(harness=harness_name, conversation_mode="resume")
    harness = harness_type(heimdall_url=catalog.url, heimdall_token="t",
                           session_root=str(tmp_path / harness_name))
    calls = []

    def invoke(question, **kwargs):
        key = "mcp_path" if harness_name == "claude_code" else "config_path"
        contents = json.loads(kwargs[key].read_text())
        env = (contents["mcpServers"]["heimdall"]["env"] if harness_name == "claude_code"
               else contents["mcp"]["heimdall"]["environment"])
        response = catalog.client.get("/api/v2/mcp/skills/alpha/", headers={
            "Authorization": "Bearer " + env["HEIMDALL_TOKEN"],
            "X-Employee-Id": env["HEIMDALL_EMPLOYEE_ID"],
            "X-Skill-Catalog-Context": env["HEIMDALL_CATALOG_CONTEXT"],
        })
        assert response.status_code == 200
        calls.append(response.json()["body"])
        assert "HEIMDALL_CATALOG_ADMIN_KEY" not in env
        return ClaudeCodeResult(answer=response.json()["body"], session_id="native-id",
                                num_turns=1, input_tokens=0, output_tokens=0,
                                cache_read_tokens=0, cache_creation_tokens=0,
                                cost_usd=0.0, duration_ms=1)

    monkeypatch.setattr(harness, "_invoke", invoke)
    for followup in (None, "native-id"):
        result = harness.run(question="test", config=config, system_prompt="system",
                             employee_id="42", b2e_session_id="session",
                             resume_session_id=followup, catalog_context=context)
        assert result.answer == "Применяй scoped.\n"
    assert calls == ["Применяй scoped.\n", "Применяй scoped.\n"]
