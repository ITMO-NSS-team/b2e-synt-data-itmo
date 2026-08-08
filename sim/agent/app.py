"""C2 — the b2e-agent service on :8082.

API:
    POST /sessions                      open a session for one employee
    POST /sessions/{id}/messages        ask a question, get an answer
    GET  /sessions/{id}                 session state and transcript
    GET  /sessions                      list/search
    POST /experiments                   batch run over a config matrix (async)
    GET  /experiments/{id}              job status
    GET  /healthz                       liveness plus the current fingerprint

Multi-tenant: every session carries its own employee identity, memory and
Heimdall permission scope. Two employees asking the same question get different
answers when their scopes differ, and that is the intended behaviour rather than
a bug to paper over.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sim import telemetry
from sim.agent.config import AgentConfig
from sim.agent.llm import ReplayMiss, build_client
from sim.agent.loop import run_turn
from sim.agent.progress import ProgressBoard
from sim.agent.prompt import DEFAULT_SYSTEM_PROMPT, render
from sim.agent.shipped import INTERACTIVE_CONFIG_REF, audit, bootstrap_if_writable
from sim.agent.store import Store
from sim.agent.tools import HeimdallTools
from sim.costguard import (
    Budget, CostCeilingExceeded, CostGuard, KillSwitchEngaged, Projection, register,
)
from sim.fingerprint import IncompleteFingerprint, RunFingerprint
from sim.registry import Registry, canonical_bytes, sha256_hex


# ------------------------------------------------------------------- schemas


class CreateSession(BaseModel):
    employee_id: str
    config_ref: str = "agent_config"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PostMessage(BaseModel):
    content: str
    question_id: str | None = None


class ExperimentRequest(BaseModel):
    """A batch over a config matrix.

    ``budget`` is required. An experiment without a declared ceiling is refused
    rather than defaulted, because a default ceiling is one nobody chose.
    """

    name: str
    questions: list[str] = Field(default_factory=list)
    basket_id: str | None = None
    employee_ids: list[str]
    config_matrix: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int | None = None
    max_usd: float | None = None
    expected_calls_per_question: int = 6


# --------------------------------------------------------------------- state


class AgentState:
    def __init__(self) -> None:
        env = os.environ
        self.heimdall_url = env.get("HEIMDALL_URL", "http://heimdall-emulator:8081")
        self.heimdall_token = env.get("HEIMDALL_TOKEN", "sim-technical-account")
        self.llm_mode = env.get("B2E_LLM_MODE", "replay")
        self.cassette_dir = env.get("B2E_CASSETTES", "cassettes")
        self.registry = Registry(env.get("B2E_REGISTRY_DB", "var/registry.db"))
        self.store = Store(env.get("B2E_AGENT_DB", "var/agent.db"))
        self.client = build_client(self.llm_mode, self.cassette_dir)
        #: In-flight turns, for surfaces with somebody waiting. Read-only to
        #: every consumer; see sim/agent/progress.py.
        self.progress = ProgressBoard()
        self._bootstrap_registry()
        self.tracing = self._configure_tracing()
        # Set before the build so /healthz has something to report even if the
        # build raises for a reason we did not anticipate.
        self.harness_status = "unknown"
        try:
            self.harness = self._build_harness()
        except Exception as exc:                              # pragma: no cover
            self.harness = None
            self.harness_status = f"failed: {type(exc).__name__}: {exc}"

    def _build_harness(self):
        """The Claude Code harness, or None if the CLI is unavailable.

        Absence is reported through /healthz rather than raised at import: the
        messages_api harness still works without the CLI, and a service that
        refuses to start would take the whole stack down over an optional
        dependency.
        """
        from shutil import which

        from sim.agent.claude_code import ClaudeCodeHarness

        env = os.environ
        claude_bin = env.get("B2E_CLAUDE_BIN", "claude")
        if which(claude_bin) is None:
            self.harness_status = f"unavailable: {claude_bin} not on PATH"
            return None

        proxy = {k: env[k] for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
                 if env.get(k)}
        self.harness_status = f"ready ({claude_bin}), proxy={'yes' if proxy else 'no'}"
        return ClaudeCodeHarness(
            heimdall_url=self.heimdall_url,
            heimdall_token=self.heimdall_token,
            bridge_path=env.get("B2E_MCP_BRIDGE", "/app/heimdall/bridge.py"),
            runner_path=env.get("B2E_SKILL_RUNNER", "/opt/skills/run"),
            claude_bin=claude_bin,
            proxy=proxy,
            # Both live on the agent's writable volume, so a conversation
            # survives a container rebuild. The CLI files session transcripts
            # per (HOME, project dir); a resumable session needs both to be the
            # same on the next turn as they were on this one.
            session_root=env.get("B2E_SESSION_ROOT", "var/sessions"),
            claude_home=env.get("B2E_CLAUDE_HOME") or None,
            timeout_seconds=int(env.get("B2E_TURN_TIMEOUT", "600")),
        )

    def _configure_tracing(self) -> str:
        """Point the tracer at Phoenix, or say plainly that it is not exporting.

        Failure here is logged and tolerated rather than fatal: an unreachable
        collector should not stop a researcher getting an answer. But it must be
        visible — a run whose spans silently went nowhere looks identical to one
        that was never made, and ``/healthz`` reporting the state is what stops
        someone spending a batch before noticing.
        """
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
        if not endpoint:
            return "disabled: PHOENIX_COLLECTOR_ENDPOINT unset"
        try:
            telemetry.configure(
                endpoint=f"{endpoint.rstrip('/')}/v1/traces",
                project_name=os.environ.get("PHOENIX_PROJECT", "b2e-sim"),
                protocol="http/protobuf",
                batch=True,
            )
            return f"exporting to {endpoint}"
        except Exception as exc:
            return f"failed: {type(exc).__name__}: {exc}"

    def _bootstrap_registry(self) -> None:
        """Seed the shipped configs, if this deployment lets us.

        It usually does not. The registry is mounted read-only here so the agent
        uid cannot reach the approval store, which means creating a config is
        somebody else's job — see ``sim/agent/shipped.py``. Reported through
        ``/healthz`` rather than raised: a missing config breaks the requests
        that name it, but taking the whole service down at boot would break the
        ones that do not.
        """
        self.registry_status = bootstrap_if_writable(self.registry)

    # --------------------------------------------------------- fingerprinting

    def skill_registry_hash(self, ref: str) -> str:
        """Digest over the skills that are actually executable right now.

        This reads the skills *table*, not the ``skill_registry`` config blob.
        The blob was written once at bootstrap as ``{"active": []}`` and nothing
        ever updated it — approving and enabling a skill goes through
        ``SkillStore.transition``, which writes the table. So the fingerprint
        field carried the SHA-256 of an empty list for the life of every
        deployment, and two runs with entirely different executable skill sets
        shared a ``condition_id`` and were treated as one experimental
        condition. The field whose whole purpose is "which skills could run"
        was a constant.
        """
        from sim.skills import SkillStore

        return SkillStore(self.registry).registry_hash()

    def emulator_condition(self) -> dict[str, Any]:
        """Ask the emulator what condition it is currently serving.

        The agent does not get to *assert* traps_enabled or the snapshot hash —
        it reads them from the service that actually holds the data. A fingerprint
        assembled from local guesses would record the condition someone intended
        rather than the one that ran.
        """
        import httpx

        try:
            r = httpx.get(f"{self.heimdall_url}/control/healthz", timeout=10.0,
                          trust_env=False)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"cannot read emulator condition from {self.heimdall_url}: "
                       f"{exc}. Refusing to start a run with a guessed fingerprint.",
            ) from exc

    def load_config(self, config_ref: str) -> tuple[Any, AgentConfig]:
        """Resolve a config ref into a dataclass, or say what is wrong with it.

        Both failures here are the same shape: the agent cannot serve, cannot
        repair, and a bare exception would surface as "Internal Server Error"
        naming neither the ref nor the remedy. So both become a 503 that says
        who holds the pen — the registry is mounted read-only to this service.
        """
        try:
            config_version, config_body = self.registry.load(config_ref)
        except KeyError as exc:
            # The realistic cause is a shipped config that no writable service
            # has created yet, so say that and say who can fix it.
            _missing, status = audit(self.registry)
            raise HTTPException(
                status_code=503,
                detail=f"config ref {config_ref!r} is not in the registry. {status}",
            ) from exc
        try:
            return config_version, AgentConfig.from_dict(config_body)
        except (ValueError, TypeError) as exc:
            # AgentConfig refuses pairs it cannot honour, and those refusals are
            # added over time: `context_strategy` other than `full` on the
            # `claude_code` harness only became one after registries in the
            # field had already been written, and `sim.registry` validates
            # nothing, so such a blob survives the upgrade untouched. It is read
            # here, on every session and every batch cell, and the only surface
            # that can rewrite it is admin-ui's /config page.
            raise HTTPException(
                status_code=503,
                detail=f"config ref {config_ref!r} is stored but not loadable: "
                       f"{exc} — repair it on the admin-ui /config page, which "
                       f"owns registry writes; this service mounts the registry "
                       f"read-only and cannot fix it.",
            ) from exc

    def build_fingerprint(self, config_ref: str) -> tuple[RunFingerprint, AgentConfig, str]:
        config_version, config = self.load_config(config_ref)
        prompt_version, prompt_body = self.registry.load(config.system_prompt_ref)
        condition = self.emulator_condition()

        try:
            fingerprint = RunFingerprint.create(
                agent_config_version=config_version.ref,
                prompt_registry_version=prompt_version.ref,
                skill_registry_hash=self.skill_registry_hash(config.skill_registry_ref),
                model_id=config.model_id,
                temperature=config.temperature,
                data_snapshot_hash=condition.get("data_snapshot_hash"),
                traps_enabled=condition.get("traps_enabled"),
                latency_profile=condition.get("latency_profile"),
                # `.get` with no default would yield None against an emulator
                # too old to report it, and None is "unset" — which would refuse
                # the run outright. An absent key genuinely means no grant, so
                # it is normalised to that rather than treated as a failure.
                hr_employee_ids=condition.get("hr_employee_ids") or [],
            )
        except IncompleteFingerprint as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return fingerprint, config, prompt_body["template"]


# ----------------------------------------------------------------------- app


def create_app(state: AgentState | None = None) -> FastAPI:
    state = state or AgentState()
    # The proxy publishes this service under /agent and strips the prefix, so
    # the app never sees it in a path. Without `root_path` the Swagger page
    # served at /agent/docs asks the browser for /openapi.json — an absolute
    # URL that misses this service entirely and lands on the proxy's fallback
    # text, which Swagger reports as "the definition does not specify a valid
    # version field". Nothing is broken except the one link that makes the page
    # useful, which is why it survived being "verified" as a 200.
    app = FastAPI(title="B2E agent", version="0.1.0",
                  root_path=os.environ.get("B2E_ROOT_PATH", ""),
                  description="The agent under test. Multi-tenant, fully traced.")
    app.state.sim = state

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        missing, _status = audit(state.registry)
        return {"status": "ok", "llm_mode": state.llm_mode,
                "heimdall": state.heimdall_url, "tracing": state.tracing,
                "harness": state.harness_status,
                # Re-read rather than served from the boot-time value: admin-ui
                # owns registry writes and starts after this service, so a ref
                # absent at boot is routinely present a few seconds later.
                "registry": state.registry_status if not missing else _status,
                "missing_configs": missing}

    @app.post("/sessions", status_code=201)
    def create_session(payload: CreateSession) -> dict[str, Any]:
        fingerprint, _config, _prompt = state.build_fingerprint(payload.config_ref)
        session_id = state.store.create_session(
            employee_id=payload.employee_id, config_ref=payload.config_ref,
            fingerprint=fingerprint.as_dict(), metadata=payload.metadata)
        return {"session_id": session_id, "employee_id": payload.employee_id,
                "fingerprint": fingerprint.as_dict(),
                "condition_id": fingerprint.condition_id}

    @app.get("/sessions")
    def list_sessions(employee_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        return {"sessions": state.store.list_sessions(
            employee_id=employee_id, limit=limit)}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session = state.store.get_session(session_id)
        if session is None:
            raise HTTPException(404, f"no session {session_id}")
        return {**session, "messages": state.store.messages(session_id)}

    @app.get("/sessions/{session_id}/progress")
    def get_progress(session_id: str) -> dict[str, Any]:
        """What the current turn is doing right now.

        Exists because a turn takes upwards of a minute and a bridge with a
        human on it needs something to show. Strictly a view: polling it cannot
        change the turn, and nothing it returns is recorded anywhere.

        404 rather than an empty body when there is no turn — "nothing is
        running" and "something is running silently" must not look alike to a
        poller.
        """
        snapshot = state.progress.get(session_id)
        if snapshot is None:
            raise HTTPException(404, f"no turn in flight for {session_id}")
        return snapshot

    @app.post("/sessions/{session_id}/messages")
    def post_message(session_id: str, payload: PostMessage) -> dict[str, Any]:
        session = state.store.get_session(session_id)
        if session is None:
            raise HTTPException(404, f"no session {session_id}")

        fingerprint, config, prompt_template = state.build_fingerprint(
            session["config_ref"])
        system_prompt = render(prompt_template, {
            "employee_id": session["employee_id"],
            "memory_block": _memory_block(config, session),
            **config.prompt_variables,
        })

        state.store.append_message(session_id=session_id, role="user",
                                   content=payload.content)
        metadata = {"question_id": payload.question_id,
                    "employee_role": session["metadata"].get("role")}

        if config.harness == "claude_code":
            result = _run_claude_code(state, session, config, system_prompt,
                                      payload.content, fingerprint, session_id,
                                      metadata)
        else:
            tools = HeimdallTools(state.heimdall_url,
                                  employee_id=session["employee_id"],
                                  token=state.heimdall_token)
            # The same switch the claude_code arm reads, so the two harnesses
            # agree on what a session *is*. They had not: messages_api always
            # replayed history while claude_code never did, which made any
            # cross-harness comparison partly a comparison of memory.
            history = (state.store.history_for_model(session_id)[:-1]
                       if config.conversation_mode == "resume" else [])
            try:
                result = run_turn(
                    question=payload.content,
                    history=history,
                    config=config, system_prompt=system_prompt,
                    client=state.client, tools=tools, fingerprint=fingerprint,
                    session_id=session_id, employee_id=session["employee_id"],
                    metadata=metadata,
                )
            except ReplayMiss as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            finally:
                tools.close()

        state.store.append_message(
            session_id=session_id, role="assistant", content=result.answer,
            trace_id=result.trace_id,
            stats={"iterations": result.iterations, "tool_calls": result.tool_calls,
                   "heimdall_calls": result.heimdall_calls,
                   "prompt_tokens": result.prompt_tokens,
                   "completion_tokens": result.completion_tokens,
                   "cost_usd": result.cost_usd, "stop_reason": result.stop_reason})

        return {
            "session_id": session_id, "answer": result.answer,
            "trace_id": result.trace_id, "stop_reason": result.stop_reason,
            "stats": {"iterations": result.iterations,
                      "tool_calls": result.tool_calls,
                      "heimdall_calls": result.heimdall_calls,
                      "prompt_tokens": result.prompt_tokens,
                      "completion_tokens": result.completion_tokens,
                      "total_tokens": result.total_tokens,
                      "cost_usd": round(result.cost_usd, 6)},
            "errors": result.errors,
            "condition_id": fingerprint.condition_id,
        }

    # -------------------------------------------------------------- batching

    @app.post("/experiments", status_code=202)
    def create_experiment(payload: ExperimentRequest, background: BackgroundTasks,
                          idempotency_key: str | None = Header(default=None)
                          ) -> JSONResponse:
        if not payload.questions and not payload.basket_id:
            raise HTTPException(422, "supply questions or a basket_id")
        if not payload.employee_ids:
            raise HTTPException(422, "supply at least one employee_id")

        try:
            budget = Budget(max_tokens=payload.max_tokens, max_usd=payload.max_usd)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        questions = payload.questions or _load_basket_questions(payload.basket_id)
        matrix = payload.config_matrix or [{}]
        runs = len(questions) * len(payload.employee_ids) * len(matrix)

        job_id, created = state.store.create_job(
            request=payload.model_dump(), idempotency_key=idempotency_key)
        if not created:
            job = state.store.get_job(job_id)
            return JSONResponse(status_code=200, content={
                "experiment_id": job_id, "status": job["status"],
                "idempotent_replay": True})

        guard = register(CostGuard(budget, experiment_id=job_id))
        _version, base_config = state.load_config("agent_config")
        # NOTE: needs recalibration, and cannot be recalibrated from the corpus
        # recorded before 2026-08-08. Until then `guard.record` was fed
        # `input_tokens`, which excludes the cached prefix, so the stored
        # per-turn token counts understate what the model read by roughly 25x
        # and no cache figure was kept anywhere. It now records the true total
        # (see ClaudeCodeResult.prompt_tokens), which means a batch can pass
        # this projection and still be stopped mid-run by its own ceiling.
        #
        # The one captured turn that does carry cache figures
        # (tests/test_sim_claude_code.py RESULT_EVENT: 30 fresh + 24 807 cache
        # read + 12 cache write over 3 iterations) suggests ~8 300 prompt
        # tokens per call rather than 6 000 — one sample, not a calibration.
        # Take the number from the first batch run under the new accounting.
        projection = Projection(
            questions=runs,
            expected_calls_per_question=payload.expected_calls_per_question,
            expected_prompt_tokens_per_call=6000,
            expected_completion_tokens_per_call=700,
            model_id=base_config.model_id)

        try:
            record = guard.check_projection(projection)
        except CostCeilingExceeded as exc:
            state.store.update_job(job_id, status="refused", error=str(exc),
                                   projection=projection.as_dict())
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        # Projection is logged before a single request is dispatched.
        state.store.update_job(job_id, status="queued", projection=record)
        background.add_task(_run_experiment, state, job_id, questions,
                            payload.employee_ids, matrix, guard)
        return JSONResponse(status_code=202, content={
            "experiment_id": job_id, "status": "queued", "runs": runs,
            "projection": record})

    @app.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        job = state.store.get_job(experiment_id)
        if job is None:
            raise HTTPException(404, f"no experiment {experiment_id}")
        guard = _guard_for(experiment_id)
        return {**job, "cost": guard.status() if guard else None}

    @app.post("/experiments/{experiment_id}/kill")
    def kill_experiment(experiment_id: str, reason: str = "operator stop") -> dict[str, Any]:
        guard = _guard_for(experiment_id)
        if guard is None:
            raise HTTPException(404, f"no running experiment {experiment_id}")
        guard.kill(reason)
        return guard.status()

    return app


def _run_claude_code(state: "AgentState", session: dict[str, Any],
                     config: AgentConfig, system_prompt: str, question: str,
                     fingerprint: RunFingerprint, session_id: str,
                     metadata: dict[str, Any]) -> "TurnResult":
    """Run one turn as a headless Claude Code session, traced like any other.

    The root span is opened here rather than inside the harness so that both
    harnesses produce the same span shape — otherwise a study comparing them
    would be comparing trace formats as much as agents.
    """
    from sim.agent.claude_code import ToolSpanRecorder, emit_spans
    from sim.agent.loop import TurnResult

    if state.harness is None:
        raise HTTPException(
            status_code=503,
            detail=f"harness=claude_code but the CLI is {state.harness_status}. "
                   f"Install it, or set harness=messages_api in the agent config.")

    # Registered before the session starts so a poller that arrives during the
    # CLI's own startup — which is seconds, before any tool runs — sees "думаю"
    # rather than a 404 it would reasonably read as "nothing happened".
    turn = state.progress.start(session_id, question)

    guard = _guard_for(metadata.get("experiment_id", "")) if metadata else None
    if guard is not None:
        # Checked before the session starts, and the spend recorded after. The
        # claude_code arm did neither, so in the DEFAULT harness the ceiling
        # could never trip and the kill switch had nothing to stop — a batch
        # reported spent_usd=0.0 while running real paid turns.
        guard.check_before_call()

    with telemetry.start_run(
        "b2e.turn", fingerprint=fingerprint, session_id=session_id,
        employee_id=session["employee_id"], metadata=metadata, question=question,
    ) as root:
        trace_id = telemetry.current_trace_id()
        # Opens a TOOL span when the CLI announces a call and closes it when the
        # result arrives, so the span carries the time the call actually took.
        # Reconstructing them after the subprocess exits — which is what this
        # did until 2026-08-08 — gives every tool span a duration of zero.
        recorder = ToolSpanRecorder(root)

        def observe(event: dict[str, Any]) -> None:
            """Both consumers of the stream, in the order that matters.

            Progress first: somebody is watching it, and a span export must not
            sit between the CLI saying what it is doing and the human seeing it.
            """
            turn.observe(event)
            recorder.observe(event)

        try:
            outcome = state.harness.run(
                question=question, config=config, system_prompt=system_prompt,
                employee_id=session["employee_id"],
                keep_stream=_keeps_raw_stream(metadata),
                b2e_session_id=session_id,
                resume_session_id=session.get("claude_session_id"),
                on_event=observe)
        except Exception:
            # A turn that dies without closing its progress leaves every poller
            # waiting on a "думаю…" that will never advance — and an open tool
            # span never exported at all.
            turn.finish(failed=True)
            recorder.finish()
            raise
        turn.finish(failed=outcome.is_error)
        recorder.finish()
        emit_spans(outcome, root=root, config=config, recorder=recorder)

    # Bound after the turn, not before: the id is what the CLI actually used,
    # which is not always the one we asked it to resume.
    if config.conversation_mode == "resume" and outcome.session_id:
        state.store.bind_claude_session(session_id, outcome.session_id)

    if guard is not None:
        guard.record(tokens=outcome.total_tokens, usd=outcome.cost_usd)

    if outcome.is_error and not outcome.answer:
        raise HTTPException(502, f"claude session failed: {outcome.error[:500]}")

    return TurnResult(
        answer=outcome.answer,
        iterations=outcome.num_turns,
        tool_calls=len(outcome.tool_calls),
        heimdall_calls=outcome.heimdall_calls,
        # The cached prefix included: `usage.input_tokens` alone is the uncached
        # remainder, which on this stack is ~30 tokens beside a ~25 000-token
        # cached prompt. See ClaudeCodeResult.prompt_tokens.
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.output_tokens,
        cost_usd=outcome.cost_usd,
        stop_reason="error" if outcome.is_error else "end_turn",
        trace_id=trace_id,
        errors=([outcome.error] if outcome.error else [])
        + [f"denied:{t}" for t in outcome.attempted_forbidden_tools],
    )


def _keeps_raw_stream(metadata: dict[str, Any] | None) -> bool:
    """Whether this turn's raw event stream is written to the session workdir.

    Kept for experiment turns only. The stream is the sole record from which a
    turn's timing can be re-derived if the instrumentation is later found to be
    wrong — and it was found to be wrong once already, which is why this exists.
    Ordinary conversation turns are not measurements and their streams carry the
    full transcript, so keeping every one of them would be an unbounded pile of
    personal data on disk for no research value.
    """
    return bool((metadata or {}).get("experiment_id"))


def _guard_for(experiment_id: str):
    from sim import costguard
    return costguard.get(experiment_id)


def _memory_block(config: AgentConfig, session: dict[str, Any]) -> str:
    """RQ3's independent variable, rendered into the prompt.

    ``none`` returns empty so the template omits the whole section — an empty
    heading would itself be a hint that memory exists.
    """
    if config.memory_strategy == "none":
        return ""
    if config.memory_strategy == "current_mart":
        return f"Идентификатор сотрудника: {session['employee_id']}."
    if config.memory_strategy == "hr_plus_external":
        return (f"Идентификатор сотрудника: {session['employee_id']}. "
                f"Профиль может присутствовать во внешней системе под другим ключом.")
    return (f"Идентификатор сотрудника: {session['employee_id']}. "
            f"Профиль может присутствовать во внешней системе под другим ключом, "
            f"а также в устаревшей реплике, покрывающей не всех сотрудников.")


def _load_basket_questions(basket_id: str | None) -> list[str]:
    """Resolve a basket selector to concrete question texts.

    ``basket_id`` selects a slice: ``all``, a family (``key_employees``), or a
    category (``prompt_injection``). It used to be passed positionally to
    ``load_basket``, whose parameters are keyword-only — so every documented
    basket run raised TypeError and returned a 500. The path had never worked.
    """
    from sim.oracle.basket import CATEGORIES, FAMILIES, load_basket

    selector = (basket_id or "all").strip()
    try:
        if selector in ("all", ""):
            questions = load_basket()
        elif selector in FAMILIES:
            questions = load_basket(families=[selector])
        elif selector in CATEGORIES:
            questions = load_basket(categories=[selector])
        else:
            raise HTTPException(
                422, f"unknown basket_id {selector!r}; expected 'all', a family "
                     f"{list(FAMILIES)}, or a category {list(CATEGORIES)}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"basket could not be loaded: {exc}") from exc

    if not questions:
        raise HTTPException(422, f"basket_id {selector!r} selected no questions")
    return [q.text for q in questions]


def _run_experiment(state: AgentState, job_id: str, questions: list[str],
                    employee_ids: list[str], matrix: list[dict[str, Any]],
                    guard: CostGuard) -> None:
    """Run the matrix sequentially.

    Sequential on purpose: 2 vCPU shared with the emulator, Phoenix and Postgres.
    Running the matrix in parallel would make every latency number a measurement
    of contention rather than of the agent.
    """
    state.store.update_job(job_id, status="running")
    results: list[dict[str, Any]] = []
    started = time.time()

    try:
        for cell in matrix:
            config_ref = cell.get("config_ref", "agent_config")
            for employee_id in employee_ids:
                for question in questions:
                    guard.check_before_call()
                    try:
                        outcome = _run_single(state, employee_id, config_ref,
                                              question, job_id)
                    except (CostCeilingExceeded, KillSwitchEngaged):
                        raise
                    except Exception as exc:
                        outcome = {"question": question, "employee_id": employee_id,
                                   "error": f"{type(exc).__name__}: {exc}"}
                    results.append(outcome)
        status = "completed"
        error = ""
    except KillSwitchEngaged as exc:
        status, error = "killed", str(exc)
    except CostCeilingExceeded as exc:
        status, error = "ceiling_reached", str(exc)
    except Exception as exc:                                  # pragma: no cover
        status, error = "failed", f"{type(exc).__name__}: {exc}"

    state.store.update_job(job_id, status=status, error=error, result={
        "runs": results, "elapsed_seconds": round(time.time() - started, 2),
        "cost": guard.status()})


def _run_single(state: AgentState, employee_id: str, config_ref: str,
                question: str, experiment_id: str) -> dict[str, Any]:
    fingerprint, config, template = state.build_fingerprint(config_ref)
    session_id = state.store.create_session(
        employee_id=employee_id, config_ref=config_ref,
        fingerprint=fingerprint.as_dict(), metadata={"experiment_id": experiment_id})
    system_prompt = render(template, {
        "employee_id": employee_id,
        "memory_block": _memory_block(config, {"employee_id": employee_id}),
        **config.prompt_variables,
    })
    started = time.perf_counter()
    if config.harness == "claude_code":
        result = _run_claude_code(
            state, {"employee_id": employee_id, "metadata": {}}, config,
            system_prompt, question, fingerprint, session_id,
            {"experiment_id": experiment_id})
    else:
        tools = HeimdallTools(state.heimdall_url, employee_id=employee_id,
                              token=state.heimdall_token)
        try:
            result = run_turn(
                question=question, history=[], config=config,
                system_prompt=system_prompt, client=state.client, tools=tools,
                fingerprint=fingerprint, session_id=session_id,
                employee_id=employee_id, guard=_guard_for(experiment_id),
                metadata={"experiment_id": experiment_id})
        finally:
            tools.close()
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    state.store.append_message(session_id=session_id, role="user", content=question)
    state.store.append_message(session_id=session_id, role="assistant",
                               content=result.answer, trace_id=result.trace_id)
    return {
        "session_id": session_id, "employee_id": employee_id,
        "config_ref": config_ref, "question": question, "answer": result.answer,
        "trace_id": result.trace_id, "condition_id": fingerprint.condition_id,
        "iterations": result.iterations, "tool_calls": result.tool_calls,
        "heimdall_calls": result.heimdall_calls,
        "total_tokens": result.total_tokens, "cost_usd": round(result.cost_usd, 6),
        "stop_reason": result.stop_reason,
        # RQ2's headline number. Nothing produced this field before, so every
        # latency percentile the research API reported was null with n=0 while
        # the whole log-normal profile machinery sat unused behind it.
        "latency_ms": latency_ms,
    }
