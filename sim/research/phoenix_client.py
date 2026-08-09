"""Minimal Phoenix REST client.

Why not the ``arize-phoenix`` Python client
-------------------------------------------
That package pulls the whole Phoenix *server* — SQLAlchemy, Strawberry GraphQL,
the UI assets — into whatever image imports it. On a box with ~2.5 GiB for the
entire stack, importing a server to make three HTTP calls is not a trade worth
making. ``arize-phoenix-otel`` (the tracing half) stays in the agent image; this
service speaks to Phoenix over its REST API.

The endpoint paths are asserted by ``make smoke`` against the running instance
rather than trusted from documentation, because a contract we do not own can
move — see ``docs/observability.md`` §6.
"""
from __future__ import annotations

from typing import Any

import httpx


class PhoenixUnavailable(RuntimeError):
    """Phoenix is not reachable. Distinguished from 'no such trace'."""


class PhoenixClient:
    def __init__(self, base_url: str, *, project: str = "b2e-sim",
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.project = project
        # trust_env=False: internal calls must not traverse an ambient proxy.
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout,
                                    trust_env=False)

    def close(self) -> None:
        self._client.close()

    def healthy(self) -> bool:
        try:
            return self._client.get("/healthz").status_code < 500
        except httpx.HTTPError:
            return False

    # ---------------------------------------------------------------- spans

    def spans_for_session(self, session_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        """All spans whose root carries this session id.

        Phoenix's span query surface has moved between versions, so this tries
        the documented v1 route and falls back rather than hard-failing: a
        researcher pulling a trace should get a clear "unavailable" instead of a
        stack trace about a route name.
        """
        try:
            response = self._client.get(
                f"/v1/projects/{self.project}/spans",
                params={"limit": limit, "filter": f"session.id == '{session_id}'"},
            )
            if response.status_code == 200:
                return _as_span_list(response.json())
            response = self._client.get("/v1/spans",
                                        params={"project_name": self.project,
                                                "limit": limit})
            if response.status_code == 200:
                spans = _as_span_list(response.json())
                return [s for s in spans
                        if _attr(s, "session.id") == session_id]
        except httpx.HTTPError as exc:
            raise PhoenixUnavailable(str(exc)) from exc
        raise PhoenixUnavailable(
            f"Phoenix returned {response.status_code} for a span query; "
            f"the REST contract may have moved (see docs/observability.md §6)")

    def latest_traces(self, *, limit: int = 100,
                      span_cap: int = 20000) -> list[dict[str, Any]]:
        """The newest ``limit`` traces, each with all of its spans.

        Two calls, not one. ``/traces?include_spans=true`` returns the spans
        without their attributes — names and timings only — which is not enough
        to show what an agent did. So the traces come first, newest by start
        time, and the spans are fetched for exactly those trace ids.

        ``span_cap`` bounds the second call. A hundred turns of this agent is on
        the order of five thousand spans, but a runaway session is the case
        where a page must stay openable, so paging stops rather than growing
        without limit. The shortfall is reported by the caller, not swallowed.
        """
        try:
            listing = self._client.get(
                f"/v1/projects/{self.project}/traces",
                params={"limit": limit, "sort": "start_time", "order": "desc"})
            if listing.status_code != 200:
                raise PhoenixUnavailable(
                    f"Phoenix returned {listing.status_code} listing traces; "
                    f"the REST contract may have moved "
                    f"(see docs/observability.md §6)")
            traces = _as_span_list(listing.json())
            if not traces:
                return []

            ids = [t.get("trace_id") for t in traces if t.get("trace_id")]
            spans: list[dict[str, Any]] = []
            cursor = None
            while len(spans) < span_cap:
                params: list[tuple[str, Any]] = [("limit", 1000)]
                params += [("trace_id", i) for i in ids]
                if cursor:
                    params.append(("cursor", cursor))
                page = self._client.get(f"/v1/projects/{self.project}/spans",
                                        params=params)
                if page.status_code != 200:
                    raise PhoenixUnavailable(
                        f"Phoenix returned {page.status_code} fetching spans")
                body = page.json()
                batch = _as_span_list(body)
                spans.extend(batch)
                cursor = body.get("next_cursor") if isinstance(body, dict) else None
                if not cursor or not batch:
                    break
        except httpx.HTTPError as exc:
            raise PhoenixUnavailable(str(exc)) from exc

        by_trace: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            trace_id = (span.get("context") or {}).get("trace_id")
            if trace_id:
                by_trace.setdefault(trace_id, []).append(span)
        for trace in traces:
            trace["spans"] = by_trace.get(trace.get("trace_id"), [])
        return traces

    # ----------------------------------------------------------- annotations

    def annotate_span(self, *, span_id: str, name: str, label: str | None,
                      score: float | None, explanation: str | None,
                      annotator: str = "researcher") -> dict[str, Any]:
        """Write feedback as a Phoenix annotation, not a bespoke table."""
        payload = {"data": [{
            "span_id": span_id,
            "name": name,
            "annotator_kind": "HUMAN",
            "result": {k: v for k, v in
                       {"label": label, "score": score,
                        "explanation": explanation}.items() if v is not None},
            "metadata": {"annotator": annotator},
        }]}
        try:
            response = self._client.post("/v1/span_annotations", json=payload)
        except httpx.HTTPError as exc:
            raise PhoenixUnavailable(str(exc)) from exc
        if response.status_code >= 400:
            raise PhoenixUnavailable(
                f"annotation rejected: {response.status_code} {response.text[:300]}")
        return {"status": "written", "span_id": span_id, "name": name}


def _as_span_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "spans", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _attr(span: dict[str, Any], key: str) -> Any:
    attributes = span.get("attributes") or {}
    if key in attributes:
        return attributes[key]
    # Phoenix sometimes returns attributes nested by dotted path.
    node: Any = attributes
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def build_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble a parent/child tree from a flat span list."""
    by_id: dict[str, dict[str, Any]] = {}
    for span in spans:
        span_id = span.get("context", {}).get("span_id") or span.get("span_id")
        if span_id:
            by_id[span_id] = {**span, "children": []}

    roots: list[dict[str, Any]] = []
    for span in by_id.values():
        parent = span.get("parent_id") or span.get("parent_span_id")
        if parent and parent in by_id:
            by_id[parent]["children"].append(span)
        else:
            roots.append(span)
    return roots
