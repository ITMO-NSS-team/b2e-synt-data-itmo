from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from sim.skill_eval.scoring.trace import retrieved_skills, root_span_id
from sim.skill_eval.types import EvalCase, SessionSpec, TurnResult


def dotenv_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip("'").strip('"')
    return ""


def proxy_client(base: str, *, public_host: str, auth: tuple[str, str] | None,
                 timeout: float) -> httpx.Client:
    parsed = urlparse(base)
    hostname = parsed.hostname or public_host or "localhost"
    header_host = public_host or hostname
    if hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = f"127.0.0.1:{parsed.port}" if parsed.port else "127.0.0.1"
        connect = urlunparse(parsed._replace(netloc=netloc))
    else:
        connect = base
    return httpx.Client(
        base_url=connect, auth=auth, verify=False, timeout=timeout,
        trust_env=False, http2=False, headers={"Host": header_host})


class StandClient:
    def __init__(
        self,
        public_url: str | None = None,
        public_host: str | None = None,
        env_file: str = "deploy/.env",
        timeout: float = 180.0,
        trace_attempts: int = 12,
        trace_backend: str = "research",
        trace_timeout: float = 300.0,
        agent_prefix: str = "/agent",
        phoenix_url: str | None = None,
        require_auth: bool = True,
    ) -> None:
        env_path = Path(env_file)
        password = (
            (os.environ.get("RESEARCHER_PASSWORD") or "").strip()
            or dotenv_value(env_path, "RESEARCHER_PASSWORD")
        )
        if require_auth and not password:
            raise RuntimeError(
                "RESEARCHER_PASSWORD is empty. Export it or put it in "
                f"{env_file}."
            )
        if trace_backend not in {"research", "phoenix"}:
            raise ValueError("trace_backend must be research or phoenix")
        self.trace_backend = trace_backend
        self.public_url = (
            public_url or dotenv_value(env_path, "PUBLIC_URL") or "https://localhost:8443"
        ).rstrip("/")
        self.public_host = public_host or dotenv_value(env_path, "PUBLIC_HOST") or urlparse(self.public_url).hostname or "localhost"
        self.timeout = timeout
        self.trace_attempts = trace_attempts
        if trace_timeout <= 0:
            raise ValueError("trace_timeout must be positive")
        self.trace_timeout = trace_timeout
        normalized_prefix = agent_prefix.strip("/")
        self.agent_prefix = f"/{normalized_prefix}" if normalized_prefix else ""
        self.phoenix_project = (
            (os.environ.get("PHOENIX_PROJECT") or "").strip()
            or dotenv_value(env_path, "PHOENIX_PROJECT")
            or "b2e-itmo"
        )
        auth = ("researcher", password) if require_auth else None
        self._client = proxy_client(
            self.public_url, public_host=self.public_host,
            auth=auth, timeout=timeout)
        self._phoenix_http = proxy_client(
            (phoenix_url or f"{self.public_url}/phoenix").rstrip("/"),
            public_host=(urlparse(phoenix_url).hostname if phoenix_url else self.public_host)
            or self.public_host,
            auth=auth, timeout=timeout)

    def run(self, case: EvalCase, spec: SessionSpec) -> TurnResult:
        started = time.monotonic()
        agent_prefix = getattr(self, "agent_prefix", "/agent")
        session = self._client.post(f"{agent_prefix}/sessions", json={
            "employee_id": spec.employee_id,
            "config_ref": spec.config_ref,
            "metadata": spec.metadata,
        })
        if session.status_code >= 400 or not session.content:
            return self._failed_turn(
                started=started,
                error=_http_error_body(
                    session, empty=f"session {session.status_code}"),
            )
        body = session.json()
        session_id = body.get("session_id")
        fingerprint = body.get("fingerprint") or {}
        live_snapshot = fingerprint.get("data_snapshot_hash")
        reply = self._client.post(
            f"{agent_prefix}/sessions/{session_id}/messages",
            json={"content": case.question},
        )
        if reply.status_code >= 400:
            return self._failed_turn(
                started=started,
                session_id=session_id,
                error=_http_error_body(
                    reply, empty=f"HTTP {reply.status_code}"),
                live_snapshot_id=live_snapshot,
                fingerprint=fingerprint,
            )
        payload = reply.json()
        stats = payload.get("stats") or {}
        heimdall_access = spec.metadata.get("heimdall_access", "enabled")
        expected_heimdall_calls = (
            int(stats.get("heimdall_calls") or 0)
            if heimdall_access != "disabled"
            else 0
        )
        trace = self._wait_trace(
            session_id, trace_id=payload.get("trace_id"),
            expected_heimdall_calls=expected_heimdall_calls,
        )
        skills = retrieved_skills(trace)
        return TurnResult(
            session_id=session_id,
            answer=str(payload.get("answer") or ""),
            stats=stats,
            trace=trace,
            error=(str(payload.get("errors")) if payload.get("errors") else None),
            retrieved_skills=skills,
            heimdall_calls=int(stats.get("heimdall_calls") or 0),
            tool_calls=int(stats.get("tool_calls") or 0),
            total_tokens=int(stats.get("total_tokens") or 0),
            latency_ms=_ms(started),
            trace_id=payload.get("trace_id") or _trace_id(trace),
            root_span_id=root_span_id(trace),
            live_snapshot_id=live_snapshot,
            fingerprint=fingerprint,
        )

    def _wait_trace(
        self, session_id: str, *, trace_id: str | None = None,
        expected_heimdall_calls: int = 0,
    ) -> dict[str, Any] | None:
        if getattr(self, "trace_backend", "research") == "phoenix":
            return self._wait_phoenix_trace(session_id, trace_id, expected_heimdall_calls)
        last: httpx.Response | None = None
        for _ in range(self.trace_attempts):
            last = self._client.get(f"/research/traces/{session_id}")
            if last.status_code == 200:
                payload = last.json()
                owned = _owned_trace(payload, session_id, trace_id)
                if owned is not None:
                    return owned
            time.sleep(1.0)
        return None

    def _wait_phoenix_trace(
        self, session_id: str, trace_id: str | None, expected_heimdall_calls: int,
    ) -> dict[str, Any] | None:
        """Poll Phoenix until this trace is complete and stable.

        Completeness is the requested ``trace_id`` (or a session-owned lookup
        when that id is absent), an ended root, and at least
        ``expected_heimdall_calls`` bridge spans. Stability is two identical
        complete reads. Empty or growing snapshots keep polling until
        ``trace_timeout``.

        Args:
            session_id: Agent session label stored on the returned payload.
            trace_id: Phoenix trace id from the agent response.
            expected_heimdall_calls: Minimum ``b2e.heimdall.endpoint`` spans.

        Returns:
            Phoenix payload, or ``None`` if the deadline elapses first.
        """
        from sim.research.phoenix_client import PhoenixClient, PhoenixUnavailable
        phoenix = PhoenixClient(
            "",
            client=self._phoenix_http,
            project=getattr(self, "phoenix_project", "b2e-itmo"),
        )
        deadline = time.monotonic() + getattr(self, "trace_timeout", 300.0)
        previous: list[dict[str, Any]] | None = None
        while True:
            complete = False
            current: list[dict[str, Any]] | None = None
            try:
                spans = (
                    phoenix.spans_for_trace(trace_id)
                    if trace_id else phoenix.spans_for_session(session_id, limit=20000)
                )
                owned = _owned_trace({"spans": spans}, session_id, trace_id)
                ended = bool(owned) and any(span.get("end_time") for span in owned["spans"])
                bridges = _bridge_count(owned["spans"]) if owned is not None else 0
                complete = (
                    owned is not None
                    and ended
                    and bridges >= expected_heimdall_calls
                )
                if complete:
                    current = _sorted_owned_spans(owned["spans"])
                    if current == previous:
                        return {
                            "session_id": session_id,
                            "spans": current,
                            "source": "phoenix",
                        }
            except (PhoenixUnavailable, ValueError):
                previous = None
            else:
                previous = current if complete else None
            if time.monotonic() >= deadline:
                return None
            time.sleep(1.0)

    def phoenix_http(self) -> httpx.Client:
        return self._phoenix_http

    def _failed_turn(
        self,
        *,
        started: float,
        error: str,
        session_id: str | None = None,
        live_snapshot_id: str | None = None,
        fingerprint: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Keep the full HTTP error and the Phoenix trace when a session exists."""
        trace = self._wait_trace(session_id) if session_id else None
        return TurnResult(
            session_id=session_id, answer="", stats={}, trace=trace,
            error=error,
            retrieved_skills=retrieved_skills(trace),
            heimdall_calls=0, tool_calls=0,
            total_tokens=0, latency_ms=_ms(started),
            trace_id=_trace_id(trace), root_span_id=root_span_id(trace),
            live_snapshot_id=live_snapshot_id,
            fingerprint=fingerprint,
        )


def _http_error_body(response: httpx.Response, *, empty: str) -> str:
    """Return status plus the complete response body, never a 400-char excerpt."""
    body = response.text or ""
    status = f"HTTP {response.status_code}"
    if not body.strip():
        return empty
    if body.startswith(status):
        return body
    return f"{status}: {body}"


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 1)


def _trace_id(trace: dict[str, Any] | None) -> str | None:
    if not trace:
        return None
    tree = trace.get("tree") or trace.get("spans") or []
    if isinstance(tree, list) and tree and isinstance(tree[0], dict):
        context = tree[0].get("context") or {}
        return (str(tree[0].get("trace_id") or "")
                or str(context.get("trace_id") or "")
                or None)
    return None


def _trace_contains_session(trace: dict[str, Any], session_id: str) -> bool:
    """Verify trace ownership instead of trusting the echoed request id."""
    from sim.research.phoenix_client import _attr
    pending: list[Any] = list(trace.get("tree") or trace.get("spans") or [])
    while pending:
        span = pending.pop()
        if not isinstance(span, dict):
            continue
        actual = _attr(span, "session.id")
        if actual == session_id:
            return True
        children = span.get("children") or []
        if isinstance(children, list):
            pending.extend(children)
    return False


def _owned_trace(
    trace: dict[str, Any], session_id: str, expected_trace_id: str | None = None,
) -> dict[str, Any] | None:
    """Drop foreign spans even when an old server returns a mixed tree."""
    def flatten(items):
        for span in items:
            if isinstance(span, dict):
                yield {key: value for key, value in span.items() if key != "children"}
                yield from flatten(span.get("children") or [])

    spans = list(flatten(trace.get("tree") or trace.get("spans") or []))
    def tid(span):
        return (span.get("context") or {}).get("trace_id") or span.get("trace_id")

    if expected_trace_id:
        matched = [span for span in spans if tid(span) == expected_trace_id]
        if not matched:
            return None
        return {"session_id": session_id, "spans": matched}
    ids = {tid(span) for span in spans if _trace_contains_session({"spans": [span]}, session_id)} - {None}
    if not ids:
        return None
    return {"session_id": session_id, "spans": [span for span in spans if tid(span) in ids]}


def _bridge_count(spans: list[dict[str, Any]]) -> int:
    """Actual Heimdall HTTP spans, not duplicated harness tool spans."""
    from sim.research.phoenix_client import _attr
    return sum(bool(_attr(span, "b2e.heimdall.endpoint")) for span in spans)


def _sorted_owned_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order spans so consecutive Phoenix reads can be compared."""
    return sorted(spans, key=lambda span: (
        not bool(span.get("end_time")),
        str((span.get("context") or {}).get("span_id") or ""),
    ))
