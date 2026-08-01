"""C5 tests: auth, CSRF, escaping, versioning, and the approval gate.

The XSS and CSRF tests are not box-ticking. The threat model records a working
chain that ends with an injected payload issuing the approve POST using the
approver's own session, so that the human "approval" never happened.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from sim.admin.app import AdminState, create_app
from sim.admin.security import CSRF_COOKIE, CSRF_FIELD
from sim.registry import Registry
from sim.skills import SkillState, SkillStore

AUTH = ("admin", "test-password")
XSS = "<script>alert('pwned')</script>"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_USER", AUTH[0])
    monkeypatch.setenv("ADMIN_PASSWORD", AUTH[1])
    monkeypatch.setenv("B2E_REGISTRY_DB", str(tmp_path / "registry.db"))
    state = AdminState()
    state.registry.commit("system_prompt", "prompt", {"template": "v1 {{ x }}"},
                          actor="bootstrap")
    state.registry.commit("agent_config", "agent",
                          __import__("sim.agent.config", fromlist=["AgentConfig"])
                          .AgentConfig().as_dict(), actor="bootstrap")
    yield TestClient(create_app(state)), state
    state.registry.close()


# ----------------------------------------------------------------- auth


def test_unauthenticated_requests_are_refused(client):
    c, _ = client
    for path in ("/", "/prompt", "/skills", "/config", "/audit"):
        assert c.get(path).status_code == 401, path


def test_wrong_password_is_refused(client):
    c, _ = client
    assert c.get("/", auth=("admin", "wrong")).status_code == 401


def test_authenticated_overview_renders(client):
    c, _ = client
    r = c.get("/", auth=AUTH)
    assert r.status_code == 200
    assert "b2e-admin" in r.text


# ------------------------------------------------------------- headers


def test_csp_forbids_script_entirely(client):
    """With no JavaScript of our own, an injected <script> is inert, not merely
    escaped."""
    c, _ = client
    csp = c.get("/", auth=AUTH).headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "script-src" not in csp     # nothing grants script
    assert "frame-ancestors 'none'" in csp


def test_clickjacking_and_sniffing_headers_present(client):
    c, _ = client
    headers = c.get("/", auth=AUTH).headers
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff"


# ---------------------------------------------------------------- csrf


def test_mutation_without_a_csrf_token_is_refused(client):
    c, _ = client
    r = c.post("/prompt", auth=AUTH, data={"template": "x", "note": ""})
    assert r.status_code == 403


def test_mutation_with_a_forged_token_is_refused(client):
    c, _ = client
    c.get("/prompt", auth=AUTH)
    r = c.post("/prompt", auth=AUTH,
               data={"template": "x", "note": "", CSRF_FIELD: "forged"})
    assert r.status_code == 403


def _token(c):
    c.get("/prompt", auth=AUTH)
    return c.cookies.get(CSRF_COOKIE)


def test_mutation_with_the_real_token_succeeds(client):
    c, state = client
    token = _token(c)
    r = c.post("/prompt", auth=AUTH, follow_redirects=False,
               data={"template": "v2 {{ x }}", "note": "test", CSRF_FIELD: token})
    assert r.status_code == 303
    assert state.registry.head("system_prompt").version == 2


# ------------------------------------------------------------ versioning


def test_saving_appends_and_never_edits_in_place(client):
    c, state = client
    token = _token(c)
    for i in range(2, 4):
        c.post("/prompt", auth=AUTH, follow_redirects=False,
               data={"template": f"v{i} {{{{ x }}}}", "note": f"n{i}",
                     CSRF_FIELD: token})
    assert state.registry.head("system_prompt").version == 3
    assert state.registry.load("system_prompt@1")[1]["template"] == "v1 {{ x }}"


def test_history_and_diff_are_shown(client):
    c, _ = client
    token = _token(c)
    c.post("/prompt", auth=AUTH, follow_redirects=False,
           data={"template": "v2 {{ x }}", "note": "changed", CSRF_FIELD: token})
    body = c.get("/prompt", auth=AUTH).text
    assert "system_prompt@2" in body
    assert "Diff vs" in body


def test_invalid_config_is_rejected_not_committed(client):
    c, state = client
    token = _token(c)
    before = state.registry.head("agent_config").version
    r = c.post("/config", auth=AUTH, follow_redirects=False,
               data={"budget_strategy": "telepathy", CSRF_FIELD: token})
    assert r.status_code == 422
    assert state.registry.head("agent_config").version == before


# ------------------------------------------------------------- escaping


def test_agent_authored_text_is_escaped_not_rendered(client):
    """heimdall's render.py emits skill text unescaped; the UI must not."""
    c, state = client
    store = SkillStore(state.registry)
    record = store.author(name=XSS, definition={},
                          code=f"# {XSS}\ndef run(payload):\n    return {{}}\n",
                          author="agent")
    body = c.get(f"/skills?show={record.code_hash}", auth=AUTH).text
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body


# -------------------------------------------------------- approval gate


def test_approval_queue_lists_uploads_not_drafts_as_approvable(client):
    c, state = client
    store = SkillStore(state.registry)
    store.author(name="from-agent", definition={},
                 code="def run(payload):\n    return {}\n", author="agent")
    body = c.get("/skills", auth=AUTH).text
    assert "Drafts (agent-authored)" in body
    assert "from-agent" in body


def test_upload_lands_in_pending_review_not_approved(client):
    c, state = client
    token = _token(c)
    c.get("/skills", auth=AUTH)
    token = c.cookies.get(CSRF_COOKIE)
    r = c.post("/skills/upload", auth=AUTH, follow_redirects=False,
               data={"name": "uploaded", "code": "def run(payload):\n    return {}\n",
                     CSRF_FIELD: token})
    assert r.status_code == 303
    store = SkillStore(state.registry)
    assert [s.state for s in store.list()] == [SkillState.PENDING_REVIEW]


def test_a_draft_cannot_be_approved_directly_through_the_ui(client):
    c, state = client
    store = SkillStore(state.registry)
    record = store.author(name="s", definition={},
                          code="def run(payload):\n    return {}\n", author="agent")
    c.get("/skills", auth=AUTH)
    token = c.cookies.get(CSRF_COOKIE)
    r = c.post("/skills/transition", auth=AUTH, follow_redirects=False,
               data={"code_hash": record.code_hash, "to": "approved",
                     CSRF_FIELD: token})
    assert r.status_code == 409
    assert store.get(record.code_hash).state is SkillState.DRAFT


def test_full_approval_path_records_the_actor(client):
    c, state = client
    store = SkillStore(state.registry)
    record = store.author(name="s", definition={},
                          code="def run(payload):\n    return {}\n", author="agent")
    c.get("/skills", auth=AUTH)
    token = c.cookies.get(CSRF_COOKIE)
    for to in ("pending_review", "approved", "active"):
        r = c.post("/skills/transition", auth=AUTH, follow_redirects=False,
                   data={"code_hash": record.code_hash, "to": to, CSRF_FIELD: token})
        assert r.status_code == 303, to
    assert store.get(record.code_hash).state is SkillState.ACTIVE
    approvals = [e for e in state.registry.audit_read()
                 if e["action"] == "skill.transition"
                 and e["detail"].get("to") == "approved"]
    assert approvals and approvals[0]["actor"] == AUTH[0]
    assert approvals[0]["detail"]["human"] is True
