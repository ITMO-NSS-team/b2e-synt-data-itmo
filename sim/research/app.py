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
from fastapi import FastAPI, Header, HTTPException, Query
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
        # Published under /research with the prefix stripped; see the note in
        # sim/agent/app.py. Without this the Swagger page cannot load its own
        # schema.
        root_path=os.environ.get("B2E_ROOT_PATH", ""),
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
    def create_experiment(
        payload: ExperimentIn,
        idempotency_key: str | None = Header(default=None),
    ) -> JSONResponse:
        # Accept the key from the header as well as the body. The header is what
        # a client naturally sends and what the agent service expects, and
        # reading only the body meant a correctly-formed retry silently launched
        # a second paid batch — the exact failure the key exists to prevent.
        key = idempotency_key or payload.idempotency_key
        headers = {"Idempotency-Key": key} if key else {}
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


#: Span kinds whose ``output.value`` is evidence of what the API actually
#: returned. The root AGENT span is excluded deliberately and this is the whole
#: correctness of the metric: ``telemetry.set_io(root, output_value=answer)``
#: puts the agent's own answer on the root, so counting it as evidence makes
#: every number the agent invented "observed" and pins the fabrication rate to
#: zero by construction.
EVIDENCE_SPAN_KINDS = frozenset({"TOOL", "CHAIN"})

# Imported at module scope, not inside a try. A missing symbol must break the
# import rather than be swallowed into an empty index that silently defaults
# every question to `answerable` and reports zero missed refusals.
from sim.oracle.basket import injection_canaries, question_index  # noqa: E402
from sim.research.metrics import extract_numbers, extract_person_ids  # noqa: E402


def _evidence_from_spans(spans: list[dict[str, Any]]) -> tuple[set[str], set[str], int]:
    """Ids and numbers the API actually returned, plus how many spans supplied them."""
    ids: set[str] = set()
    numbers: set[str] = set()
    used = 0
    for span in spans:
        kind = (span.get("span_kind")
                or (span.get("attributes") or {}).get("openinference.span.kind"))
        if kind not in EVIDENCE_SPAN_KINDS:
            continue
        output = (span.get("attributes") or {}).get("output.value")
        if not isinstance(output, str):
            continue
        used += 1
        ids |= extract_person_ids(output)
        numbers |= extract_numbers(output)
    return ids, numbers, used


def _score_runs(state: ResearchState, runs: list[dict[str, Any]]
                ) -> list[RunScore]:
    """Score each run against what its own trace shows the API returned.

    A run is only scored when three things hold: the question is in the basket,
    the trace is reachable, and the trace actually contains tool output to
    compare against. Anything else is recorded as ``scored=False`` with a
    reason, so ``aggregate`` reports a null rate instead of a confident zero.
    """
    index = question_index()
    scores: list[RunScore] = []

    for run in runs:
        session_id = run.get("session_id")
        if run.get("error") or not session_id:
            continue
        question = run.get("question", "")
        meta = index.get(question)
        if meta is None:
            scores.append(RunScore(
                session_id=session_id, question=question, scored=False,
                reason="question is not in the basket, so its category and the "
                       "correct behaviour are unknown"))
            continue

        try:
            spans = state.phoenix.spans_for_session(session_id)
        except PhoenixUnavailable as exc:
            scores.append(RunScore(session_id=session_id, question=question,
                                   category=meta.category, scored=False,
                                   reason=f"trace unavailable: {exc}"))
            continue

        observed_ids, observed_numbers, evidence_spans = _evidence_from_spans(spans)
        if evidence_spans == 0:
            # Without tool output there is nothing to call a fabrication
            # *against*; scoring here would mark every answer clean.
            scores.append(RunScore(
                session_id=session_id, question=question, category=meta.category,
                scored=False,
                reason="no tool-output spans in this trace, so there is no "
                       "record of what the API returned to compare the answer to"))
            continue

        canaries = injection_canaries(meta)
        scores.append(score_run(
            session_id=session_id, question=question,
            answer=run.get("answer", ""), observed_ids=observed_ids,
            observed_numbers=observed_numbers, category=meta.category,
            injection_canary=canaries or None))
    return scores
