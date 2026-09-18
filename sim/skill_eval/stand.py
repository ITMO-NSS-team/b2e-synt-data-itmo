from __future__ import annotations

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


def proxy_client(base: str, *, public_host: str, auth: tuple[str, str],
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
        public_url: str = "https://localhost:8443",
        public_host: str = "localhost",
        env_file: str = "deploy/.env",
        timeout: float = 180.0,
        trace_attempts: int = 12,
    ) -> None:
        env_path = Path(env_file)
        password = dotenv_value(env_path, "RESEARCHER_PASSWORD")
        if not password:
            raise RuntimeError(
                "RESEARCHER_PASSWORD is empty. Put it in deploy/.env.")
        self.public_url = (
            dotenv_value(env_path, "PUBLIC_URL") or public_url
        ).rstrip("/")
        self.public_host = dotenv_value(env_path, "PUBLIC_HOST") or public_host
        self.timeout = timeout
        self.trace_attempts = trace_attempts
        auth = ("researcher", password)
        self._client = proxy_client(
            self.public_url, public_host=self.public_host,
            auth=auth, timeout=timeout)
        self._phoenix_http = proxy_client(
            f"{self.public_url}/phoenix", public_host=self.public_host,
            auth=auth, timeout=timeout)

    def run(self, case: EvalCase, spec: SessionSpec) -> TurnResult:
        started = time.monotonic()
        session = self._client.post("/agent/sessions", json={
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
            f"/agent/sessions/{session_id}/messages",
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
        trace = self._wait_trace(session_id)
        skills = retrieved_skills(trace)
        return TurnResult(
            session_id=session_id,
            answer=str(payload.get("answer") or ""),
            stats=stats,
            trace=trace,
            error=None,
            retrieved_skills=skills,
            heimdall_calls=int(stats.get("heimdall_calls") or 0),
            tool_calls=int(stats.get("tool_calls") or 0),
            total_tokens=int(stats.get("total_tokens") or 0),
            latency_ms=_ms(started),
            trace_id=_trace_id(trace),
            root_span_id=root_span_id(trace),
            live_snapshot_id=live_snapshot,
            fingerprint=fingerprint,
        )

    def _wait_trace(self, session_id: str) -> dict[str, Any] | None:
        last: httpx.Response | None = None
        for _ in range(self.trace_attempts):
            last = self._client.get(f"/research/traces/{session_id}")
            if last.status_code == 200:
                payload = last.json()
                if _trace_contains_session(payload, session_id):
                    return payload
            time.sleep(1.0)
        return None

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
    pending: list[Any] = list(trace.get("tree") or trace.get("spans") or [])
    while pending:
        span = pending.pop()
        if not isinstance(span, dict):
            continue
        attrs = span.get("attributes") or {}
        actual = attrs.get("session.id")
        if actual is None and isinstance(attrs.get("session"), dict):
            actual = attrs["session"].get("id")
        if actual == session_id:
            return True
        children = span.get("children") or []
        if isinstance(children, list):
            pending.extend(children)
    return False
