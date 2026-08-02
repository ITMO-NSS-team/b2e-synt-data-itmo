"""C1 — assembly of the Heimdall emulator service on :8081.

Adds to ``heimdall.create_app`` the four things an experiment needs:

1. **Latency** drawn per endpoint and per request shape (``sim.latency``).
2. **Per-employee permission scoping**, so 403 is a real outcome.
3. **Runtime toggles** for the two independent variables.
4. **Recruitment RPC stubs** in the RpcEnvelope shape.

The body-buffering middleware is written at ASGI level rather than with
``BaseHTTPMiddleware``. Starlette's request body is a one-shot stream: reading it
in a ``BaseHTTPMiddleware`` dispatch consumes it, and the downstream route then
receives an empty body. Buffering and replaying the receive channel is the only
way to both inspect a request and let it proceed unchanged.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse

from sim import latency as latency_mod
from sim.emulator.config import EmulatorConfig, SnapshotUnavailable
from sim.emulator.identity import AccessDenied, IdentityIndex, enforce, restrict
from sim.emulator.rpc import rpc_router

#: Header carrying the employee the agent is acting for. Separate from the
#: bearer token, which authenticates the *service*, not the human — the same
#: split the production system uses.
ACTING_EMPLOYEE_HEADER = "x-employee-id"

#: Prefixes refused outright before routing.
#:
#: ``heimdall``'s dev router (``/api/v2/dev/*``) creates, reads and deletes skill
#: files over plain HTTP with **no authentication at all** — it does not use
#: ``_require_bearer``. Two of its handlers also derive the write path from
#: fields inside the uploaded file, so a crafted upload writes outside the
#: skills root. Reachable by the agent, that is a direct route from "the agent
#: produced text" to "the registry serves it as instructions".
#:
#: This is enforced in middleware rather than by removing routes. Filtering
#: ``app.router.routes`` does not work on this FastAPI version: included routers
#: are stored as ``_IncludedRouter`` objects with no ``.path`` attribute, so a
#: path-based filter silently matches nothing and *appears* to succeed. A guard
#: on the request path is version-independent and, unlike route surgery, is
#: directly testable.
BLOCKED_PREFIXES = ("/api/v2/dev/",)


def _classify(path: str) -> str:
    """Map a URL path to a latency endpoint key.

    Keys are logical operation names so that renaming a route does not silently
    drop an endpoint back to the default distribution.
    """
    if "/mcp/query/" in path:
        return "mcp_query"
    if "/mcp/models/" in path:
        return "describe_model" if path.rstrip("/").count("/") > 4 else "list_models"
    if "/mcp/docs" in path:
        return "get_docs"
    if "/overview" in path:
        return "overview"
    if path.rstrip("/").endswith("/skills"):
        return "find_skills"
    if "/skills/" in path:
        return "get_skill"
    if "/rpc/" in path:
        return "rpc"
    if path.startswith("/api/v1/") and path.count("/") >= 4:
        return "mcp_query"
    return "default"


class EmulatorState:
    """Mutable runtime state: which condition the emulator is currently serving.

    Quirks are mutated in place and the snapshot object is swapped. Rebuilding
    the whole FastAPI app on every toggle would be cleaner in principle but
    costs a catalogue reload, and the admin UI toggles these between runs.
    """

    def __init__(self, config: EmulatorConfig) -> None:
        self.config = config
        self.traps_enabled = config.traps_enabled
        self.latency_profile = config.latency_profile
        #: Set by create_app. Needed to decide whether a model even has a person
        #: column before a row-scope predicate is appended to a query.
        self.catalog = None
        self._identities: dict[bool, IdentityIndex] = {}
        #: (endpoint, canonical request) -> times seen, for latency occurrence.
        self._occurrence: dict[tuple[str, str], int] = {}

    # --------------------------------------------------------------- identity

    def identity_index(self) -> IdentityIndex:
        key = self.traps_enabled
        if key not in self._identities:
            root = self.config.snapshot_for(key)
            self._identities[key] = IdentityIndex(
                root, hr_employee_ids=self.config.hr_employee_ids)
        return self._identities[key]

    # ---------------------------------------------------------------- latency

    def next_occurrence(self, endpoint: str, body: Any) -> int:
        try:
            canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            canon = repr(body)
        key = (endpoint, canon)
        seen = self._occurrence.get(key, 0)
        self._occurrence[key] = seen + 1
        return seen

    def scope_column(self, schema: str, model: str) -> str | None:
        """The person-key column this model can be restricted on, if any.

        Returns None for reference data (org units, positions, dictionaries),
        which has no person dimension and must not be filtered — appending a
        predicate on a column that does not exist would turn every such query
        into an ``unknown-column`` error instead of an authorised read.
        """
        if self.catalog is None:
            return None
        found = self.catalog.get(schema, model)
        if found is None:
            return None
        names = {c.name if hasattr(c, "name") else str(c) for c in found.columns}
        for candidate in ("person_id", "employee_id"):
            if candidate in names:
                return candidate
        return None

    def fingerprint_fragment(self) -> dict[str, Any]:
        return {
            "traps_enabled": self.traps_enabled,
            "latency_profile": self.latency_profile,
            "data_snapshot_hash": self.config.snapshot_id(self.traps_enabled),
        }


# ------------------------------------------------------------------ middleware


class SimulationMiddleware:
    """Buffers the body, enforces scope, then injects latency after the response.

    Latency is applied *after* the handler runs, not before, because the delay
    depends on how much data came back. A wide result set is slow because it is
    wide; sleeping first would have to guess.
    """

    def __init__(self, app, state: EmulatorState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        if not self.state.config.enable_dev_router and _is_blocked(path):
            # 404, not 403: a disabled surface should not advertise that it
            # exists. There is nothing here to negotiate access to.
            await _error_response(404, "model-not-found",
                                  "endpoint not available in this deployment",
                                  "the skill dev API is disabled; use the admin UI "
                                  "approval queue")(scope, receive, send)
            return

        body = b""
        if method in ("POST", "PUT", "PATCH"):
            chunks = []
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    chunks.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        break
                elif message["type"] == "http.disconnect":
                    return
            body = b"".join(chunks)

            replayed = {"done": False}

            async def replay():
                if replayed["done"]:
                    return {"type": "http.disconnect"}
                replayed["done"] = True
                return {"type": "http.request", "body": body, "more_body": False}

            receive = replay

        parsed: Any = None
        if body:
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = None

        # ---- scope enforcement, before any work is done
        if isinstance(parsed, dict) and _needs_scope(path):
            denial = self._check_scope(scope, path, parsed)
            if denial is not None:
                await denial(scope, receive, send)
                return
            # Refusing named people is not enough; a query with no person filter
            # would otherwise return the whole mart. Rewrite the body so the
            # restriction is part of the query the engine actually runs.
            restricted = self._restrict_body(scope, path, parsed)
            if restricted is not None:
                parsed = restricted
                body = json.dumps(restricted, ensure_ascii=False).encode("utf-8")

                async def replay_restricted(_state={"done": False}):
                    if _state["done"]:
                        return {"type": "http.disconnect"}
                    _state["done"] = True
                    return {"type": "http.request", "body": body,
                            "more_body": False}

                receive = replay_restricted

        endpoint = _classify(path)
        started = time.perf_counter()

        # The response is BUFFERED rather than streamed through, because in ASGI
        # a `http.response.body` with more_body=False completes the response.
        # Sleeping after `await self.app(...)` therefore delays only this
        # coroutine and not the caller — measured at 4ms for `degraded` against
        # 3ms for `instant`, i.e. RQ2's independent variable did nothing at all
        # while the fingerprint dutifully labelled the runs as different
        # conditions. The delay has to happen before the first byte goes out.
        buffered: list[dict] = []
        body_chunks: list[bytes] = []

        async def buffer(message):
            buffered.append(message)
            if message["type"] == "http.response.body":
                body_chunks.append(message.get("body", b""))

        await self.app(scope, receive, buffer)

        profile = latency_mod.get_profile(self.state.latency_profile)
        n_rows = _count_rows(b"".join(body_chunks))
        n_columns = _count_columns(parsed)
        target_ms = latency_mod.sample_ms(
            profile, endpoint, request=parsed,
            occurrence=self.state.next_occurrence(endpoint, parsed),
            n_columns=n_columns, n_rows=n_rows,
        )
        spent_ms = (time.perf_counter() - started) * 1000.0
        remaining = (target_ms - spent_ms) / 1000.0
        if remaining > 0:
            await asyncio.sleep(remaining)

        # Report what was actually injected, so a p95 built from these traces can
        # be separated into simulated and real time instead of blending them.
        for message in buffered:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"x-b2e-injected-latency-ms",
                                f"{max(0.0, target_ms - spent_ms):.1f}".encode()))
                message = {**message, "headers": headers}
            await send(message)

    # ------------------------------------------------------------------ scope

    def _restrict_body(self, scope, path: str, parsed: dict[str, Any]):
        """Row-scope the query, or None when no restriction applies."""
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        employee = headers.get(ACTING_EMPLOYEE_HEADER)
        if not employee:
            return None
        schema, model = _target_model(path, parsed)
        if schema is None or model is None:
            return None
        try:
            employee_scope = self.state.identity_index().scope_for(employee)
        except (AccessDenied, SnapshotUnavailable):
            return None
        if employee_scope.sees_everything:
            return None
        column = self.state.scope_column(schema, model)
        if column is None:
            return None
        return restrict(employee_scope, parsed, column=column)

    def _check_scope(self, scope, path: str, parsed: dict[str, Any]):
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        employee = headers.get(ACTING_EMPLOYEE_HEADER)
        if not employee:
            return _error_response(
                403, "forbidden",
                f"missing {ACTING_EMPLOYEE_HEADER} header",
                "every data request acts for a specific employee; anonymous "
                "access is not modelled")

        schema, model = _target_model(path, parsed)
        if schema is None:
            return None

        try:
            index = self.state.identity_index()
            employee_scope = index.scope_for(employee)
            enforce(employee_scope, schema, model, parsed)
        except AccessDenied as exc:
            status = 403 if exc.code in ("forbidden", "auth-failed") else 400
            return _error_response(status, exc.code, exc.detail, exc.hint)
        except SnapshotUnavailable as exc:
            return _error_response(503, "internal-error", str(exc), "")
        return None


def _is_blocked(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in BLOCKED_PREFIXES)


def _needs_scope(path: str) -> bool:
    return path.startswith("/api/v1/") and "/models/" not in path and "/docs" not in path


def _target_model(path: str, parsed: dict[str, Any]) -> tuple[str | None, str | None]:
    """Which model is this request about? Body first, then the REST path."""
    schema = parsed.get("schema")
    model = parsed.get("logic_model")
    if isinstance(schema, str) and isinstance(model, str):
        return schema, model
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 4 and parts[0] == "api" and parts[1] == "v1":
        return parts[2], parts[3]
    return None, None


def _count_columns(parsed: Any) -> int:
    """Requested column count. ``["*"]`` is the expensive case RQ2 cares about."""
    if not isinstance(parsed, dict):
        return 0
    columns = parsed.get("columns")
    if not isinstance(columns, list):
        return 0
    if any(c == "*" for c in columns):
        return 642                      # widest mart in the catalogue
    return len(columns)


def _count_rows(chunk: bytes) -> int:
    try:
        payload = json.loads(chunk)
    except (ValueError, UnicodeDecodeError):
        return 0
    if isinstance(payload, dict):
        for key in ("rows", "data", "items", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return 0


def _error_response(status: int, code: str, detail: str, hint: str):
    body = json.dumps({"code": code, "detail": detail, "hint": hint},
                      ensure_ascii=False).encode("utf-8")

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    return app


# ----------------------------------------------------------------------- app


def create_app(config: EmulatorConfig | None = None) -> FastAPI:
    from heimdall.app import create_app as create_heimdall_app
    from heimdall.engine.quirks import ALL_QUIRKS, Quirks

    config = config or EmulatorConfig.from_env()
    state = EmulatorState(config)

    quirks = Quirks(enabled=set(ALL_QUIRKS) if config.traps_enabled else set())
    snapshot_root = config.snapshot_for(config.traps_enabled)

    app = create_heimdall_app(snapshot_root, config.catalog_path,
                              config.skills_root, quirks=quirks)
    app.title = "Heimdall Emulator (simulation)"
    app.state.sim = state
    state.catalog = getattr(app.state, "heimdall", None) and app.state.heimdall.catalog
    app.state.quirks = quirks

    app.include_router(rpc_router(state))
    app.include_router(_control_router(state, quirks))
    app.add_middleware(SimulationMiddleware, state=state)
    return app


def _control_router(state: EmulatorState, quirks) -> APIRouter:
    """Operator surface. Bound to the compose network, never exposed publicly."""
    from heimdall.engine.quirks import ALL_QUIRKS

    router = APIRouter(prefix="/control", tags=["control"])

    @router.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", **state.fingerprint_fragment()}

    @router.get("/config")
    def get_config() -> dict[str, Any]:
        return {
            "traps_enabled": state.traps_enabled,
            "latency_profile": state.latency_profile,
            "quirks_enabled": sorted(quirks.enabled),
            "snapshot": str(state.config.snapshot_for(state.traps_enabled)),
            "data_snapshot_hash": state.config.snapshot_id(state.traps_enabled),
            "hr_employee_ids": list(state.config.hr_employee_ids),
            "dev_router_enabled": state.config.enable_dev_router,
        }

    @router.put("/config")
    def put_config(payload: dict[str, Any]) -> dict[str, Any]:
        if "latency_profile" in payload:
            profile = str(payload["latency_profile"])
            if profile not in latency_mod.PROFILES:
                return JSONResponse(status_code=422, content={
                    "code": "request-validation-error",
                    "detail": f"unknown latency profile {profile!r}",
                    "hint": f"one of {latency_mod.PROFILES}"})
            state.latency_profile = profile

        if "traps_enabled" in payload:
            want = bool(payload["traps_enabled"])
            try:
                state.config.snapshot_for(want)
            except SnapshotUnavailable as exc:
                # Refuse rather than serve traps-on data labelled traps-off.
                return JSONResponse(status_code=409, content={
                    "code": "internal-error", "detail": str(exc),
                    "hint": "build the traps-off corpus before switching"})
            state.traps_enabled = want
            quirks.enabled = set(ALL_QUIRKS) if want else set()

        return get_config()

    @router.get("/latency-profiles")
    def latency_profiles() -> list[dict[str, Any]]:
        return latency_mod.describe_profiles()

    @router.get("/identities")
    def identities(n: int = 5) -> list[dict[str, Any]]:
        return state.identity_index().sample_identities(n)

    @router.get("/scope")
    def scope(x_employee_id: str = Header(...)) -> dict[str, Any]:
        try:
            found = state.identity_index().scope_for(x_employee_id)
        except AccessDenied as exc:
            return JSONResponse(status_code=403, content={
                "code": exc.code, "detail": exc.detail, "hint": exc.hint})
        return {
            "employee_id": found.employee_id, "person_id": found.person_id,
            "role": found.role, "unit_id": found.unit_id,
            "visible_people": ("all" if found.sees_everything
                               else len(found.visible_person_ids)),
        }

    return router
