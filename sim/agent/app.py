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
from sim.agent.prompt import DEFAULT_SYSTEM_PROMPT, render
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
        self._bootstrap_registry()
        self.tracing = self._configure_tracing()

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
        """Seed version 1 of every config so an untouched deployment still has a
        version to name in the fingerprint."""
        if self.registry.head("system_prompt") is None:
            self.registry.commit("system_prompt", "prompt",
                                 {"template": DEFAULT_SYSTEM_PROMPT},
                                 actor="bootstrap", note="shipped default")
        if self.registry.head("agent_config") is None:
            self.registry.commit("agent_config", "agent", AgentConfig().as_dict(),
                                 actor="bootstrap", note="shipped default")
        if self.registry.head("skill_registry") is None:
            self.registry.commit("skill_registry", "skills", {"active": []},
                                 actor="bootstrap", note="empty registry")

    # --------------------------------------------------------- fingerprinting

    def skill_registry_hash(self, ref: str) -> str:
        version, body = self.registry.load(ref)
        active = sorted(body.get("active", []))
        return "sha256:" + sha256_hex(canonical_bytes(active))

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

    def build_fingerprint(self, config_ref: str) -> tuple[RunFingerprint, AgentConfig, str]:
        config_version, config_body = self.registry.load(config_ref)
        config = AgentConfig.from_dict(config_body)
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
            )
        except IncompleteFingerprint as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return fingerprint, config, prompt_body["template"]


# ----------------------------------------------------------------------- app


def create_app(state: AgentState | None = None) -> FastAPI:
    state = state or AgentState()
    app = FastAPI(title="B2E agent", version="0.1.0",
                  description="The agent under test. Multi-tenant, fully traced.")
    app.state.sim = state

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "llm_mode": state.llm_mode,
                "heimdall": state.heimdall_url, "tracing": state.tracing}

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

        tools = HeimdallTools(state.heimdall_url,
                              employee_id=session["employee_id"],
                              token=state.heimdall_token)
        try:
            state.store.append_message(session_id=session_id, role="user",
                                       content=payload.content)
            result = run_turn(
                question=payload.content,
                history=state.store.history_for_model(session_id)[:-1],
                config=config, system_prompt=system_prompt,
                client=state.client, tools=tools, fingerprint=fingerprint,
                session_id=session_id, employee_id=session["employee_id"],
                metadata={"question_id": payload.question_id,
                          "employee_role": session["metadata"].get("role")},
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
        base_config = AgentConfig.from_dict(
            state.registry.load("agent_config")[1])
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
    try:
        from sim.oracle.basket import load_basket
    except ImportError:
        raise HTTPException(503, "question basket is not available in this build")
    return [q.text for q in load_basket(basket_id)]


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
    }
