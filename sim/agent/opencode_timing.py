"""Collect OpenCode's native AI SDK timings without exporting duplicate spans.

Pinned OpenCode 1.18.32 exports OTLP/HTTP JSON when experimental.openTelemetry
is enabled. Keep only numeric timings and IDs; discard prompts, outputs and logs.
"""
from __future__ import annotations

import json
import math
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from sim.agent.claude_code import ClaudeCodeResult


SOURCE = "opencode_ai_sdk_otlp"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return None


class NativeTimings:
    def __init__(self) -> None:
        self.endpoint = ""
        self.requests: dict[str, dict[str, Any]] = {}

    def ingest(self, payload: dict) -> None:
        for resource in payload.get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    if span.get("name") != "ai.streamText.doStream":
                        continue
                    attrs = {a["key"]: next(iter(a.get("value", {}).values()), None)
                             for a in span.get("attributes", []) if "key" in a}
                    identity = f'{span.get("traceId", "")}:{span.get("spanId", "")}'
                    start, end = int(span.get("startTimeUnixNano", 0)), int(span.get("endTimeUnixNano", 0))
                    if identity == ":" or start <= 0 or end < start:
                        continue
                    self.requests[identity] = {
                        "start": start, "end": end,
                        "session": attrs.get("session.id") or attrs.get("ai.telemetry.metadata.sessionId"),
                        "first_chunk": _number(attrs.get("ai.response.msToFirstChunk")),
                        "finish": _number(attrs.get("ai.response.msToFinish")),
                    }

    def apply(self, result: ClaudeCodeResult, *, started_ns: int) -> None:
        # Match the model call announced by stdout, not background title/summary
        # requests or history from another turn. Ambiguous matches stay unknown.
        matched = []
        used = set()
        for call in result.llm_calls:
            candidates = [(key, request) for key, request in self.requests.items()
                          if request["session"] == result.session_id
                          and started_ns <= request["start"] <= call.started_ns <= request["end"]
                          and request["start"] <= call.ended_ns]
            if len(candidates) != 1 or candidates[0][0] in used:
                continue
            key, request = candidates[0]
            used.add(key)
            call.api_duration_ms = request["finish"]
            call.ttft_stream_ms = request["first_chunk"]
            call.timing_source = SOURCE
            matched.append(request)
        result.timing_source = SOURCE
        result.timing_calls = len(matched)
        # Partial totals would look artificially fast. Leave unavailable unless
        # every foreground call has a corresponding completed native request.
        if matched and len(matched) == len(result.llm_calls):
            if all(r["finish"] is not None for r in matched):
                result.api_duration_ms = sum(r["finish"] for r in matched)
            first = min(matched, key=lambda r: r["start"])
            result.ttft_stream_ms = first["first_chunk"]
            result.time_to_request_ms = (first["start"] - started_ns) / 1_000_000


@contextmanager
def collect_native_timings() -> Iterator[NativeTimings]:
    """Private per-invocation receiver, in the CLI subprocess's network namespace.

    Loopback means the agent container (also on remote deployments), NOT the
    machine viewing Phoenix. This is not a remote/public OTLP collector.
    """
    capture = NativeTimings()
    path = "/" + uuid.uuid4().hex

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:
            self.connection.settimeout(5)
            if self.path not in (path + "/v1/traces", path + "/v1/logs"):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = self.rfile.read(size)
                if self.path.endswith("/traces"):
                    capture.ingest(json.loads(body))
            except (ValueError, TypeError, KeyError, AttributeError, OSError):
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # Wait for in-flight exports before applying the captured measurements.
    server.daemon_threads = False
    capture.endpoint = f"http://127.0.0.1:{server.server_port}{path}"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield capture
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
