"""C5 — admin UI on :8084.

Server-rendered, no framework, no JavaScript. That is a security choice as much
as a size one: with no script of our own, the CSP can forbid script entirely, and
an injected ``<script>`` in an agent-authored skill description is inert rather
than merely escaped.

One page is an exception and is treated as one. The *Trace explorer* renders
client-side, so it is served under a policy that permits inline script and
nothing else — see ``EXPLORER_CONTENT_SECURITY_POLICY`` for what still holds it
in, and why a payload of agent-authored text cannot escape its container.

Panels: system prompt editor with diff and version history; skill registry with
the approval queue; model parameters; traps and latency toggles; a session list
that deep-links into Phoenix; the trace explorer, which shows the last hundred
turns in full — fingerprint, iterations, reasoning, tool calls and the Heimdall
layer — read live from Phoenix; and the audit log.

Every mutation is versioned and appended. Nothing is edited in place.
"""
from __future__ import annotations

import difflib
import html
import json
import os
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse, JSONResponse, RedirectResponse, Response,
)

from sim import traceview
from sim.admin.security import (
    CSRF_COOKIE, CSRF_FIELD, headers_for, issue_csrf, require_admin,
    verify_csrf,
)
from sim.agent.config import AgentConfig
from sim.agent.shipped import bootstrap
from sim.registry import Registry
from sim.skills import SkillState, SkillStore, static_check

CSS = """
:root{--fg:#1b1b1f;--bg:#fbfbfd;--mut:#6b6b76;--line:#e2e2e8;--ok:#0a7d32;--no:#b3261e}
@media(prefers-color-scheme:dark){:root{--fg:#e6e6ea;--bg:#151518;--mut:#9a9aa5;--line:#2c2c33}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,sans-serif;color:var(--fg);background:var(--bg)}
header{border-bottom:1px solid var(--line);padding:.75rem 1.25rem;display:flex;gap:1.25rem;align-items:baseline;flex-wrap:wrap}
header b{font-size:1rem}
nav a{color:var(--fg);text-decoration:none;margin-right:1rem;font-size:.9rem}
nav a:hover{text-decoration:underline}
main{padding:1.25rem;max-width:1100px}
h2{font-size:1.1rem;margin:1.5rem 0 .5rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.5rem 0}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.84rem}
pre{background:rgba(128,128,128,.09);padding:.7rem;border-radius:6px;overflow-x:auto;max-width:100%}
textarea{width:100%;min-height:16rem;font-family:ui-monospace,monospace;font-size:.84rem;padding:.6rem;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg)}
input,select{padding:.35rem .5rem;border:1px solid var(--line);border-radius:5px;background:transparent;color:var(--fg)}
button{padding:.4rem .8rem;border:1px solid var(--line);border-radius:5px;background:transparent;color:var(--fg);cursor:pointer;font-size:.88rem}
button.primary{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.mut{color:var(--mut)}
.ok{color:var(--ok)}.no{color:var(--no)}
.badge{font-size:.75rem;padding:.1rem .45rem;border:1px solid var(--line);border-radius:99px}
.row{display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-end;margin:.5rem 0}
.ins{color:var(--ok)}.del{color:var(--no)}
.warn{border-left:3px solid var(--no);padding:.5rem .8rem;margin:.8rem 0;background:rgba(179,38,30,.06)}
"""

#: Navigation, as *relative* references.
#:
#: The proxy publishes this app under /admin and strips the prefix, so the app
#: cannot know its own mount point — and a root-anchored link to /config
#: therefore sent the browser to https://host/config, which is not a route and
#: renders as the proxy's fallback text. Every link in the UI was broken; only
#: typing /admin/config by hand worked.
#:
#: Relative references fix it without the app having to know anything. From
#: /admin/config the base is /admin/, so "config" resolves to /admin/config;
#: running bare on :8084 the base is /, and the same string resolves to
#: /config. One spelling, correct under both mounts.
#:
#: `.` rather than "" for Overview: an empty href means "this page" and would
#: make the tab a no-op on every page but the root.
NAV = [(".", "Overview"), ("prompt", "System prompt"), ("skills", "Skills"),
       ("config", "Model"), ("emulator", "Traps &amp; latency"),
       ("traces", "Traces"), ("explorer", "Trace explorer"), ("audit", "Audit")]

#: How many traces the explorer tab loads. A hundred turns of this agent is on
#: the order of five thousand spans; the page holds them because the
#: reconstructed prompts — which are quadratic in a turn's length — are left out
#: of the live view and named as left out.
EXPLORER_TRACES = 100


def esc(value: Any) -> str:
    """Escape everything rendered. Skill text is attacker-influenced."""
    return html.escape(str(value), quote=True)


def page(title: str, body: str, csrf: str) -> str:
    nav = "".join(f'<a href="{href}">{label}</a>' for href, label in NAV)
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)} · b2e-admin</title><style>{CSS}</style></head>"
            f"<body><header><b>b2e-admin</b><nav>{nav}</nav></header>"
            f"<main>{body}</main></body></html>")


def csrf_input(token: str) -> str:
    return f'<input type=hidden name={CSRF_FIELD} value="{esc(token)}">'


def _config_for_form(body_json: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Stored agent config as a plain dict, loadable or not.

    ``AgentConfig`` gains constraints over time — ``context_strategy`` other than
    ``full`` now requires ``harness='messages_api'``, and that pair was legal (and
    offered by the dropdown on this very page) until it wasn't. The registry is a
    dumb versioned store that validates nothing, so a blob written before the
    constraint stays exactly where it is. Constructing it here would 500 both the
    GET and the POST, and the only surface that can repair the blob is the form
    those two handlers render — an unrepairable loop.

    So: merge over the shipped defaults instead. Every known key is then present
    for rendering whatever the blob is missing, unknown keys ride along so a save
    still refuses them by name, and the caller gets the exception text to show.
    """
    merged = {**AgentConfig().as_dict(), **body_json}
    try:
        AgentConfig.from_dict(body_json)
    except (ValueError, TypeError) as exc:
        return merged, str(exc)
    return merged, None


class AdminState:
    def __init__(self) -> None:
        env = os.environ
        self.registry = Registry(env.get("B2E_REGISTRY_DB", "var/registry.db"))
        # This service holds the only read-write mount of the registry, which
        # makes it the only one that can create a shipped config. b2e-agent
        # reads the same file read-only by design (the agent uid must not reach
        # the approval store), so if the seeding did not happen here it would
        # not happen at all — and the first Telegram message would 503 on a ref
        # that nothing in the stack was able to write.
        self.seeded = bootstrap(self.registry)
        self.skills = SkillStore(self.registry)
        self.emulator_url = env.get("HEIMDALL_URL", "http://heimdall-emulator:8081")
        #: Where to send the browser. Public, and usually not resolvable here.
        self.phoenix_url = env.get("PHOENIX_PUBLIC_URL",
                                   env.get("PHOENIX_URL", "http://localhost:6006"))
        #: Where *this process* talks to Phoenix. Kept apart from the public URL
        #: because they are different addresses for different callers, and using
        #: the public one for a server-side call fails in exactly the deployment
        #: where it matters — behind the proxy, where it is an https host this
        #: container has no route to.
        self.phoenix_api_url = env.get("PHOENIX_URL", "http://phoenix:6006")
        self.phoenix_project = env.get("PHOENIX_PROJECT", "b2e-sim")
        self.agent_url = env.get("B2E_AGENT_URL", "http://b2e-agent:8082")
        self._http = httpx.Client(timeout=20.0, trust_env=False)
        self._phoenix: Any = None

    @property
    def phoenix(self) -> Any:
        """Built on first use, not at boot.

        The admin UI has to come up whether or not Phoenix is running — it is
        where an operator goes to find out why something is broken, and a
        constructor that reached the network would make the diagnostic page the
        second casualty of the outage it exists to explain.
        """
        if self._phoenix is None:
            from sim.research.phoenix_client import PhoenixClient
            self._phoenix = PhoenixClient(self.phoenix_api_url,
                                          project=self.phoenix_project,
                                          timeout=60.0)
        return self._phoenix

    def emulator_config(self) -> dict[str, Any]:
        try:
            return self._http.get(f"{self.emulator_url}/control/config").json()
        except httpx.HTTPError as exc:
            return {"error": str(exc)}

    def set_emulator_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self._http.put(f"{self.emulator_url}/control/config", json=payload)
            return {"status": r.status_code, "body": r.json()}
        except httpx.HTTPError as exc:
            return {"status": 503, "body": {"detail": str(exc)}}

    def sessions(self, limit: int = 40) -> list[dict[str, Any]]:
        try:
            return self._http.get(f"{self.agent_url}/sessions",
                                  params={"limit": limit}).json().get("sessions", [])
        except httpx.HTTPError:
            return []


def create_app(state: AdminState | None = None) -> FastAPI:
    state = state or AdminState()
    app = FastAPI(title="B2E admin", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.sim = state

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for key, value in headers_for(request.url.path).items():
            response.headers[key] = value
        return response

    def _csrf(request: Request, response: Response) -> str:
        token = request.cookies.get(CSRF_COOKIE) or issue_csrf()
        response.set_cookie(CSRF_COOKIE, token, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", path="/")
        return token

    # ------------------------------------------------------------ overview

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request, response: Response,
                 actor: str = Depends(require_admin)) -> HTMLResponse:
        token = _csrf(request, response)
        emu = state.emulator_config()
        pending = state.skills.list(SkillState.PENDING_REVIEW)
        drafts = state.skills.list(SkillState.DRAFT)
        active = state.skills.list(SkillState.ACTIVE)

        rows = "".join(
            f"<tr><td>{esc(name)}</td><td><code>{esc(v.ref)}</code></td>"
            f"<td class=mut>{esc(v.actor)}</td></tr>"
            for name in state.registry.names()
            for v in [state.registry.head(name)] if v)

        body = f"""
<h2>Current condition</h2>
<table>
<tr><th>traps_enabled</th><td><code>{esc(emu.get('traps_enabled'))}</code></td></tr>
<tr><th>latency_profile</th><td><code>{esc(emu.get('latency_profile'))}</code></td></tr>
<tr><th>data_snapshot_hash</th><td><code>{esc(emu.get('data_snapshot_hash'))}</code></td></tr>
<tr><th>skill_registry_hash</th><td><code>{esc(state.skills.registry_hash())}</code></td></tr>
</table>

<h2>Config heads</h2>
<table><tr><th>name</th><th>version</th><th>last actor</th></tr>{rows}</table>

<h2>Skills</h2>
<p><a href="skills">{len(pending)} awaiting review</a> ·
{len(drafts)} draft (agent-authored) · {len(active)} active</p>
<p class=mut>Signed in as {esc(actor)}.</p>
"""
        return HTMLResponse(page("Overview", body, token),
                            headers=dict(response.headers))

    # -------------------------------------------------------------- prompt

    @app.get("/prompt", response_class=HTMLResponse)
    def prompt_editor(request: Request, response: Response, version: int | None = None,
                      actor: str = Depends(require_admin)) -> HTMLResponse:
        token = _csrf(request, response)
        history = state.registry.history("system_prompt")
        if not history:
            raise HTTPException(503, "no system_prompt in the registry yet")
        head = history[0]
        chosen = (state.registry.get_version("system_prompt", version)
                  if version else head)
        if chosen is None:
            raise HTTPException(404, f"no system_prompt@{version}")
        text = state.registry.get_json(chosen.hash)["template"]

        diff_html = ""
        if len(history) > 1:
            previous = state.registry.get_json(history[1].hash)["template"]
            diff = difflib.unified_diff(previous.splitlines(), text.splitlines(),
                                        fromfile=history[1].ref, tofile=chosen.ref,
                                        lineterm="")
            lines = []
            for line in diff:
                cls = "ins" if line.startswith("+") else \
                      "del" if line.startswith("-") else "mut"
                lines.append(f'<span class="{cls}">{esc(line)}</span>')
            diff_html = f"<h2>Diff vs {esc(history[1].ref)}</h2><pre>" \
                        + "\n".join(lines) + "</pre>"

        versions = "".join(
            f'<tr><td><a href="prompt?version={v.version}"><code>{esc(v.ref)}</code></a></td>'
            f'<td class=mut>{esc(v.actor)}</td><td class=mut>{esc(v.note)}</td>'
            f'<td><code>{esc(v.hash[:12])}</code></td></tr>' for v in history)

        body = f"""
<h2>System prompt — editing {esc(chosen.ref)}</h2>
<p class=mut>Saving appends a new version. Nothing is edited in place, so every
trace recorded against an older version can still resolve it.</p>
<form method=post action="prompt">
{csrf_input(token)}
<textarea name=template>{esc(text)}</textarea>
<div class=row><input name=note placeholder="why this change" size=48>
<button class=primary type=submit>Save as new version</button></div>
</form>
{diff_html}
<h2>Version history</h2>
<table><tr><th>version</th><th>actor</th><th>note</th><th>sha256</th></tr>{versions}</table>
"""
        return HTMLResponse(page("System prompt", body, token),
                            headers=dict(response.headers))

    @app.post("/prompt")
    def save_prompt(request: Request, template: str = Form(...),
                    note: str = Form(""), csrf_token: str = Form(None),
                    actor: str = Depends(require_admin)) -> RedirectResponse:
        verify_csrf(request, csrf_token)
        state.registry.commit("system_prompt", "prompt", {"template": template},
                              actor=actor, note=note)
        return RedirectResponse("prompt", status_code=303)

    # -------------------------------------------------------------- skills

    @app.get("/skills", response_class=HTMLResponse)
    def skills_page(request: Request, response: Response, show: str | None = None,
                    actor: str = Depends(require_admin)) -> HTMLResponse:
        token = _csrf(request, response)

        detail = ""
        if show:
            record = state.skills.get(show)
            if record is None:
                raise HTTPException(404, "no such skill hash")
            code = (state.skills.code(show) or b"").decode("utf-8", "replace")
            checks = static_check(code)
            undeclared = checks.get("undeclared") or []
            warning = (f'<div class=warn>Imports outside the declarable set: '
                       f'<code>{esc(", ".join(undeclared))}</code>. '
                       f'The sandbox blocks these at runtime, but their presence '
                       f'means the author expected capabilities they will not get.'
                       f'</div>' if undeclared else "")
            actions = ""
            if record.state is SkillState.DRAFT:
                actions = _action_form(token, show, "pending_review",
                                       "Submit for review")
            elif record.state is SkillState.PENDING_REVIEW:
                actions = (_action_form(token, show, "approved", "Approve",
                                        primary=True)
                           + _action_form(token, show, "rejected", "Reject"))
            elif record.state is SkillState.APPROVED:
                actions = (_action_form(token, show, "active", "Enable", primary=True)
                           + _action_form(token, show, "retired", "Retire"))
            elif record.state is SkillState.ACTIVE:
                actions = (_action_form(token, show, "approved", "Disable")
                           + _action_form(token, show, "retired", "Retire"))

            detail = f"""
<h2>{esc(record.name)} <span class=badge>{esc(record.state.value)}</span></h2>
<table>
<tr><th>sha256</th><td><code>{esc(record.code_hash)}</code></td></tr>
<tr><th>author</th><td>{esc(record.author)}</td></tr>
<tr><th>entrypoint</th><td>{'yes' if checks.get('ok') else esc(checks.get('error'))}</td></tr>
<tr><th>imports</th><td><code>{esc(", ".join(checks.get('imports') or []) or '—')}</code></td></tr>
</table>
{warning}
<p class=mut>These are the exact bytes that will execute. Approval attaches to
the digest above, so any edit produces a different hash with no approval.</p>
<pre>{esc(code)}</pre>
<div class=row>{actions}</div>
"""

        def table(state_filter: SkillState, title: str) -> str:
            records = state.skills.list(state_filter)
            if not records:
                return f"<h2>{esc(title)}</h2><p class=mut>none</p>"
            rows = "".join(
                f'<tr><td><a href="skills?show={esc(r.code_hash)}">{esc(r.name)}</a></td>'
                f'<td><code>{esc(r.code_hash[:12])}</code></td>'
                f'<td class=mut>{esc(r.author)}</td></tr>' for r in records)
            return (f"<h2>{esc(title)}</h2><table>"
                    f"<tr><th>name</th><th>sha256</th><th>author</th></tr>"
                    f"{rows}</table>")

        upload = f"""
<h2>Upload a skill</h2>
<p class=mut>Uploaded skills enter as <code>pending_review</code>. Upload is not
approval.</p>
<form method=post action="skills/upload">
{csrf_input(token)}
<div class=row><input name=name placeholder="skill name" required></div>
<textarea name=code placeholder="def run(payload): ..."></textarea>
<div class=row><button type=submit>Upload for review</button></div>
</form>
"""
        body = (detail
                + table(SkillState.PENDING_REVIEW, "Approval queue")
                + table(SkillState.DRAFT, "Drafts (agent-authored)")
                + table(SkillState.ACTIVE, "Active")
                + table(SkillState.APPROVED, "Approved but disabled")
                + upload)
        return HTMLResponse(page("Skills", body, token),
                            headers=dict(response.headers))

    @app.post("/skills/upload")
    def upload_skill(request: Request, name: str = Form(...), code: str = Form(...),
                     csrf_token: str = Form(None),
                     actor: str = Depends(require_admin)) -> RedirectResponse:
        verify_csrf(request, csrf_token)
        record = state.skills.upload(name=name, definition={"name": name},
                                     code=code, actor=actor)
        return RedirectResponse(f"skills?show={record.code_hash}", status_code=303)

    @app.post("/skills/transition")
    def transition_skill(request: Request, code_hash: str = Form(...),
                         to: str = Form(...), csrf_token: str = Form(None),
                         actor: str = Depends(require_admin)) -> RedirectResponse:
        verify_csrf(request, csrf_token)
        try:
            # is_human_action=True is justified here and only here: this handler
            # is reachable only behind Basic auth, with a matching CSRF token,
            # from a form that displayed the exact bytes and their digest.
            state.skills.transition(code_hash, SkillState(to), actor=actor,
                                    is_human_action=True)
        except Exception as exc:
            raise HTTPException(409, str(exc)) from exc
        return RedirectResponse(f"skills?show={code_hash}", status_code=303)

    # --------------------------------------------------------------- config

    @app.get("/config", response_class=HTMLResponse)
    def config_page(request: Request, response: Response,
                    actor: str = Depends(require_admin)) -> HTMLResponse:
        token = _csrf(request, response)
        version, body_json = state.registry.load("agent_config")
        current, load_error = _config_for_form(body_json)
        history = state.registry.history("agent_config")
        rows = "".join(
            f'<tr><td><code>{esc(v.ref)}</code></td><td class=mut>{esc(v.actor)}</td>'
            f'<td class=mut>{esc(v.note)}</td></tr>' for v in history)

        def field(name: str, value: Any, options: tuple[str, ...] | None = None) -> str:
            if options:
                opts = "".join(
                    f'<option{" selected" if o == value else ""}>{esc(o)}</option>'
                    for o in options)
                return (f"<tr><th>{esc(name)}</th><td>"
                        f"<select name={name}>{opts}</select></td></tr>")
            return (f"<tr><th>{esc(name)}</th><td>"
                    f'<input name={name} value="{esc(value)}"></td></tr>')

        from sim.agent.config import (
            BUDGET_STRATEGIES, CODE_EXECUTION_MODES, CONTEXT_STRATEGIES,
            HARNESSES, MEMORY_STRATEGIES,
        )

        stale_warning = ""
        if load_error:
            stale_warning = (
                '<div class=warn><b>The stored config is not loadable.</b> '
                f'{esc(load_error)} The form below shows it merged over the '
                'shipped defaults so you can correct the offending pair and '
                'save a legal version; until you do, every session and every '
                'batch cell against this ref fails.</div>')

        code_warning = ""
        if current["code_execution"] == "allowed":
            code_warning = (
                '<div class=warn><b>Code execution is ON.</b> This is the rival '
                'arm, not the default. Two things are true while it is set: the '
                'agent may run arbitrary Python, so no tool policy constrains '
                'which files it opens — the comparison is only sound where the '
                'corpus is not mounted into the agent container; and the agent '
                'process holds the model credential, which arbitrary code can '
                'read. Use a dedicated low-quota token for this arm.</div>')

        body = f"""
<h2>Model parameters — {esc(version.ref)}</h2>
{stale_warning}
{code_warning}
<form method=post action="config">
{csrf_input(token)}
<table>
{field("model_id", current["model_id"])}
{field("temperature", current["temperature"])}
{field("max_output_tokens", current["max_output_tokens"])}
{field("context_window_tokens", current["context_window_tokens"])}
{field("max_tool_iterations", current["max_tool_iterations"])}
{field("harness", current["harness"], HARNESSES)}
{field("code_execution", current["code_execution"], CODE_EXECUTION_MODES)}
{field("budget_strategy", current["budget_strategy"], BUDGET_STRATEGIES)}
{field("context_strategy", current["context_strategy"], CONTEXT_STRATEGIES)}
{field("memory_strategy", current["memory_strategy"], MEMORY_STRATEGIES)}
</table>
<div class=row><input name=note placeholder="why this change" size=48>
<button class=primary type=submit>Save as new version</button></div>
</form>
<p class=mut>The context window is a config value, not a constant: hardcoding
200 000 becomes a lie the moment the model changes, and the failure is silent.</p>
<p class=mut><code>code_execution=allowed</code> lifts the project's premise so the
rival hypothesis can be measured rather than assumed. It requires
<code>harness=claude_code</code>; the messages_api loop has no tool that can run
code, so the combination is refused rather than silently mislabelled.</p>
<p class=mut><code>context_strategy=windowed</code> and <code>summarised</code> are
implemented only by the messages_api loop's packer, so they require
<code>harness=messages_api</code>. Under <code>claude_code</code> the CLI owns its
own context window and the field was read by nothing at all — three values, three
condition_ids, one behaviour. Both dropdowns stay fully populated because harness
is edited on this same form: switch the pair together and the save succeeds;
submit an incoherent pair and you get a 422 naming it.</p>
<h2>Version history</h2>
<table><tr><th>version</th><th>actor</th><th>note</th></tr>{rows}</table>
"""
        return HTMLResponse(page("Model", body, token),
                            headers=dict(response.headers))

    @app.post("/config")
    async def save_config(request: Request, csrf_token: str = Form(None),
                          actor: str = Depends(require_admin)) -> RedirectResponse:
        form = await request.form()
        verify_csrf(request, csrf_token)
        updated, _load_error = _config_for_form(state.registry.load("agent_config")[1])
        for key in ("model_id", "budget_strategy", "context_strategy",
                    "memory_strategy", "harness", "code_execution"):
            if key in form:
                updated[key] = str(form[key])
        for key in ("temperature",):
            if key in form:
                updated[key] = float(form[key])
        for key in ("max_output_tokens", "context_window_tokens",
                    "max_tool_iterations"):
            if key in form:
                updated[key] = int(form[key])
        try:
            AgentConfig.from_dict(updated)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        state.registry.commit("agent_config", "agent", updated, actor=actor,
                              note=str(form.get("note", "")))
        return RedirectResponse("config", status_code=303)

    # ------------------------------------------------------------- emulator

    @app.get("/emulator", response_class=HTMLResponse)
    def emulator_page(request: Request, response: Response,
                      actor: str = Depends(require_admin)) -> HTMLResponse:
        token = _csrf(request, response)
        cfg = state.emulator_config()
        body = f"""
<h2>Traps and latency</h2>
<p class=mut>These are the two independent variables. Both appear in the run
fingerprint, so switching one makes subsequent runs a different condition.</p>
<form method=post action="emulator">
{csrf_input(token)}
<div class=row>
<label>traps_enabled
<select name=traps_enabled>
<option value=true{' selected' if cfg.get('traps_enabled') else ''}>true</option>
<option value=false{'' if cfg.get('traps_enabled') else ' selected'}>false</option>
</select></label>
<label>latency_profile
<select name=latency_profile>
{"".join(f'<option{" selected" if p == cfg.get("latency_profile") else ""}>{p}</option>' for p in ("instant", "realistic", "degraded"))}
</select></label>
<button class=primary type=submit>Apply</button>
</div>
</form>
<pre>{esc(json.dumps(cfg, ensure_ascii=False, indent=1))}</pre>
<p class=mut>Turning traps off requires a traps-off corpus. If it has not been
built, the emulator refuses with 409 rather than serving traps-on data under a
traps-off label.</p>
"""
        return HTMLResponse(page("Traps &amp; latency", body, token),
                            headers=dict(response.headers))

    @app.post("/emulator")
    def set_emulator(request: Request, traps_enabled: str = Form(...),
                     latency_profile: str = Form(...), csrf_token: str = Form(None),
                     actor: str = Depends(require_admin)) -> RedirectResponse:
        verify_csrf(request, csrf_token)
        result = state.set_emulator_config({
            "traps_enabled": traps_enabled == "true",
            "latency_profile": latency_profile})
        state.registry.audit_write(
            actor=actor, action="emulator.config", target="heimdall-emulator",
            detail={"requested": {"traps_enabled": traps_enabled,
                                  "latency_profile": latency_profile},
                    "result": result})
        return RedirectResponse("emulator", status_code=303)

    # --------------------------------------------------------------- traces

    @app.get("/traces", response_class=HTMLResponse)
    def traces_page(request: Request, response: Response,
                    actor: str = Depends(require_admin)) -> HTMLResponse:
        token = _csrf(request, response)
        sessions = state.sessions()
        rows = "".join(
            f'<tr><td><code>{esc(s["id"])}</code></td>'
            f'<td>{esc(s.get("employee_id"))}</td>'
            f'<td><code>{esc((s.get("fingerprint") or {}).get("latency_profile"))}</code></td>'
            f'<td><code>{esc((s.get("fingerprint") or {}).get("traps_enabled"))}</code></td>'
            f'<td><a href="{esc(state.phoenix_url)}/projects" rel=noreferrer>Phoenix</a></td>'
            f'</tr>' for s in sessions)
        body = f"""
<h2>Recent sessions</h2>
<p class=mut>Spans live in Phoenix. This is a deep link, not a second copy.</p>
<table><tr><th>session</th><th>employee</th><th>latency</th><th>traps</th><th></th></tr>
{rows or '<tr><td colspan=5 class=mut>no sessions yet</td></tr>'}</table>
"""
        return HTMLResponse(page("Traces", body, token),
                            headers=dict(response.headers))

    # ------------------------------------------------------- trace explorer

    @app.get("/explorer", response_class=HTMLResponse)
    def explorer_page(request: Request, response: Response,
                      limit: int = EXPLORER_TRACES,
                      input_messages: bool = False,
                      actor: str = Depends(require_admin)) -> HTMLResponse:
        """The standalone trace viewer, served live off Phoenix.

        The same page `build_viewer.py` produces from an export file, and the
        same assembly code behind it — see `sim/traceview.py` for why that is
        not two implementations. What differs is only the source and the
        freshness.
        """
        import datetime as dt

        token = _csrf(request, response)
        limit = max(1, min(int(limit), 500))
        try:
            document = traceview.from_phoenix(
                state.phoenix, limit=limit, project=state.phoenix_project,
                source=f"Arize Phoenix REST at {state.phoenix_api_url}, "
                       f"project '{state.phoenix_project}'",
                include_input_messages=bool(input_messages),
                exported_at=dt.datetime.now(dt.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception as exc:
            # Never a stack trace on an operator's screen: this page failing
            # says something about Phoenix, and that is the sentence to print.
            body = f"""
<h2>Trace explorer</h2>
<div class=warn><b>Phoenix is not answering.</b> {esc(type(exc).__name__)}:
{esc(str(exc)[:400])}</div>
<p class=mut>The explorer reads {esc(state.phoenix_api_url)} directly. Spans
live there; this page is a view, not a second copy.</p>
"""
            return HTMLResponse(page("Trace explorer", body, token),
                                headers=dict(response.headers))

        if not document["traces"]:
            body = """
<h2>Trace explorer</h2>
<p class=mut>No traces recorded yet. Ask the agent something and reload.</p>
"""
            return HTMLResponse(page("Trace explorer", body, token),
                                headers=dict(response.headers))

        return HTMLResponse(traceview.render(document),
                            headers=dict(response.headers))

    @app.get("/explorer.json")
    def explorer_json(request: Request, response: Response,
                      limit: int = EXPLORER_TRACES,
                      input_messages: bool = False,
                      actor: str = Depends(require_admin)) -> JSONResponse:
        """The same document as JSON.

        So the page's own 'Load JSON…' button has something to eat, and so a
        researcher can pull the live corpus without shelling into a container.
        """
        import datetime as dt

        limit = max(1, min(int(limit), 500))
        try:
            document = traceview.from_phoenix(
                state.phoenix, limit=limit, project=state.phoenix_project,
                include_input_messages=bool(input_messages),
                exported_at=dt.datetime.now(dt.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception as exc:
            raise HTTPException(503, f"Phoenix unavailable: {exc}") from exc
        return JSONResponse(document)

    # ---------------------------------------------------------------- audit

    @app.get("/audit", response_class=HTMLResponse)
    def audit_page(request: Request, response: Response,
                   actor: str = Depends(require_admin)) -> HTMLResponse:
        token = _csrf(request, response)
        import datetime as dt
        rows = "".join(
            f'<tr><td class=mut>{esc(dt.datetime.fromtimestamp(e["ts"]).isoformat(timespec="seconds"))}</td>'
            f'<td>{esc(e["actor"])}</td><td><code>{esc(e["action"])}</code></td>'
            f'<td><code>{esc(e["target"])}</code></td>'
            f'<td class=mut>{esc(json.dumps(e["detail"], ensure_ascii=False)[:160])}</td></tr>'
            for e in state.registry.audit_read(limit=200))
        body = f"""
<h2>Audit log</h2>
<p class=mut>Append-only. Every config commit and every lifecycle transition
lands here with actor and timestamp.</p>
<table><tr><th>when</th><th>actor</th><th>action</th><th>target</th><th>detail</th></tr>
{rows}</table>
"""
        return HTMLResponse(page("Audit", body, token),
                            headers=dict(response.headers))

    return app


def _action_form(token: str, code_hash: str, to: str, label: str,
                 primary: bool = False) -> str:
    return (f'<form method=post action="skills/transition" style="display:inline">'
            f'{csrf_input(token)}'
            f'<input type=hidden name=code_hash value="{esc(code_hash)}">'
            f'<input type=hidden name=to value="{esc(to)}">'
            f'<button type=submit class="{"primary" if primary else ""}">'
            f'{esc(label)}</button></form> ')
