"""C4 — research-api on :8083.

    GET  /traces/{session_id}          full span tree, JSON
    GET  /sessions                     list/search
    POST /feedback                     write a Phoenix annotation
    POST /experiments                  launch a batch over a config matrix
    GET  /experiments/{id}             status and metrics vs the oracle
    GET  /experiments/{id}/export      Parquet or JSONL
    GET  /openapi.json                 published spec

A thin layer. Experiments are dispatched to the agent service rather than run
here, and spans are read from Phoenix rather than stored here — the spec is
explicit that this must not become a second tracing system, and the fastest way
to become one is to start keeping "just a small copy" of the spans.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from sim.research.metrics import RunScore, aggregate, score_run
from sim.research.phoenix_client import (
    PhoenixClient, PhoenixUnavailable, build_tree,
)


class FeedbackIn(BaseModel):
    """Feedback at session OR individual-response level.

    Exactly one target is required. Accepting both would silently pick one and
    attach the annotation somewhere the researcher did not intend.
    """

    session_id: str | None = None
    span_id: str | None = None
    label: Literal["like", "dislike"]
    explanation: str | None = None
    annotator: str = "researcher"


class ExperimentIn(BaseModel):
    name: str
    questions: list[str] = Field(default_factory=list)
    basket_id: str | None = None
    employee_ids: list[str]
    config_matrix: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int | None = None
    max_usd: float | None = None
    idempotency_key: str | None = None


class ResearchState:
    def __init__(self) -> None:
        env = os.environ
        self.agent_url = env.get("B2E_AGENT_URL", "http://b2e-agent:8082")
        self.phoenix_url = env.get("PHOENIX_URL", "http://phoenix:6006")
        self.project = env.get("PHOENIX_PROJECT", "b2e-sim")
        self.phoenix = PhoenixClient(self.phoenix_url, project=self.project)
        self._http = httpx.Client(timeout=60.0, trust_env=False)

    def agent(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, f"{self.agent_url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise HTTPException(
                503, f"b2e-agent unreachable at {self.agent_url}: {exc}") from exc


def create_app(state: ResearchState | None = None) -> FastAPI:
    state = state or ResearchState()
    app = FastAPI(
        title="B2E research API",
        version="0.1.0",
        description=(
            "Read/write API for researchers. Thin layer over Phoenix and the run "
            "store. Traces live in Phoenix; feedback is stored as Phoenix "
            "annotations, not in a bespoke table."
        ),
    )
    app.state.sim = state

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "phoenix": state.phoenix.healthy(),
                "agent_url": state.agent_url}

    # ------------------------------------------------------------- traces

    @app.get("/traces/{session_id}", summary="Full span tree for one session")
    def get_trace(session_id: str) -> dict[str, Any]:
        try:
            spans = state.phoenix.spans_for_session(session_id)
        except PhoenixUnavailable as exc:
            raise HTTPException(503, f"Phoenix unavailable: {exc}") from exc
        if not spans:
            raise HTTPException(404, f"no spans for session {session_id}")
        return {"session_id": session_id, "span_count": len(spans),
                "tree": build_tree(spans)}

    @app.get("/sessions", summary="List or search sessions")
    def list_sessions(employee_id: str | None = None,
                      limit: int = Query(50, le=500)) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if employee_id:
            params["employee_id"] = employee_id
        response = state.agent("GET", "/sessions", params=params)
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.text[:500])
        return response.json()

    # ----------------------------------------------------------- feedback

    @app.post("/feedback", status_code=201,
              summary="Write feedback as a Phoenix annotation")
    def post_feedback(payload: FeedbackIn) -> dict[str, Any]:
        if bool(payload.session_id) == bool(payload.span_id):
            raise HTTPException(
                422, "supply exactly one of session_id or span_id: session-level "
                     "and response-level feedback attach to different targets")

        target = payload.span_id
        if target is None:
            try:
                spans = state.phoenix.spans_for_session(payload.session_id or "")
            except PhoenixUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
            roots = build_tree(spans)
            if not roots:
                raise HTTPException(404, f"no spans for session {payload.session_id}")
            target = (roots[0].get("context", {}).get("span_id")
                      or roots[0].get("span_id"))

        try:
            written = state.phoenix.annotate_span(
                span_id=target, name="user_feedback", label=payload.label,
                score=1.0 if payload.label == "like" else 0.0,
                explanation=payload.explanation, annotator=payload.annotator)
        except PhoenixUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        return {**written, "level": "span" if payload.span_id else "session"}

    # -------------------------------------------------------- experiments

    @app.post("/experiments", status_code=202, summary="Launch a batch run")
    def create_experiment(payload: ExperimentIn) -> JSONResponse:
        headers = ({"Idempotency-Key": payload.idempotency_key}
                   if payload.idempotency_key else {})
        body = payload.model_dump(exclude={"idempotency_key"})
        response = state.agent("POST", "/experiments", json=body, headers=headers)
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.text[:1000])
        return JSONResponse(status_code=response.status_code, content=response.json())

    @app.get("/experiments/{experiment_id}",
             summary="Status and metrics vs the oracle question bank")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        response = state.agent("GET", f"/experiments/{experiment_id}")
        if response.status_code == 404:
            raise HTTPException(404, f"no experiment {experiment_id}")
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.text[:500])

        job = response.json()
        runs = (job.get("result") or {}).get("runs", [])
        scores = _score_runs(state, runs)
        return {
            "experiment_id": experiment_id,
            "status": job.get("status"),
            "error": job.get("error"),
            "projection": job.get("projection"),
            "cost_guard": job.get("cost"),
            "metrics": aggregate(runs, scores),
            "conditions": sorted({r.get("condition_id") for r in runs
                                  if r.get("condition_id")}),
        }

    @app.get("/experiments/{experiment_id}/export",
             summary="Export runs as Parquet or JSONL")
    def export_experiment(experiment_id: str,
                          fmt: Literal["jsonl", "parquet"] = "jsonl"
                          ) -> StreamingResponse:
        response = state.agent("GET", f"/experiments/{experiment_id}")
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.text[:500])
        runs = (response.json().get("result") or {}).get("runs", [])
        if not runs:
            raise HTTPException(404, "experiment has no runs to export")

        if fmt == "jsonl":
            body = "\n".join(json.dumps(r, ensure_ascii=False, default=str)
                             for r in runs)
            return StreamingResponse(
                io.BytesIO(body.encode("utf-8")), media_type="application/x-ndjson",
                headers={"Content-Disposition":
                         f'attachment; filename="{experiment_id}.jsonl"'})

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise HTTPException(
                503, "pyarrow is not installed; use fmt=jsonl") from exc

        flat = [{k: (json.dumps(v, ensure_ascii=False, default=str)
                     if isinstance(v, (dict, list)) else v)
                 for k, v in run.items()} for run in runs]
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(flat), buffer)
        buffer.seek(0)
        return StreamingResponse(
            buffer, media_type="application/vnd.apache.parquet",
            headers={"Content-Disposition":
                     f'attachment; filename="{experiment_id}.parquet"'})

    return app


def _score_runs(state: ResearchState, runs: list[dict[str, Any]]
                ) -> list[RunScore]:
    """Score each run against what its own trace shows the API returned.

    Returns an empty list when nothing can be scored, so ``aggregate`` reports a
    null hallucination rate with a reason rather than a zero.
    """
    try:
        from sim.oracle.basket import question_index
        index = question_index()
    except Exception:
        index = {}

    scores: list[RunScore] = []
    for run in runs:
        if run.get("error") or not run.get("session_id"):
            continue
        question = run.get("question", "")
        meta = index.get(question)
        category = getattr(meta, "category", "answerable") if meta else "answerable"
        canary = getattr(meta, "injection_canary", None) if meta else None

        observed_ids: set[str] = set()
        observed_numbers: set[str] = set()
        try:
            spans = state.phoenix.spans_for_session(run["session_id"])
            for span in spans:
                output = (span.get("attributes") or {}).get("output.value")
                if isinstance(output, str):
                    from sim.research.metrics import extract_numbers, extract_person_ids
                    observed_ids |= extract_person_ids(output)
                    observed_numbers |= extract_numbers(output)
        except PhoenixUnavailable:
            continue

        scores.append(score_run(
            session_id=run["session_id"], question=question,
            answer=run.get("answer", ""), observed_ids=observed_ids,
            observed_numbers=observed_numbers, category=category,
            injection_canary=canary))
    return scores
