from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EvalCase:
    case_id: str
    category: str
    question: str
    runtime_actor_employee_id: str
    expected_skill: str | None
    expected_skill_kind: str | None
    gold: dict[str, Any]
    snapshot_id: str | None
    business_task: dict[str, Any]
    raw: dict[str, Any]


@dataclass(slots=True)
class SessionSpec:
    employee_id: str
    config_ref: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class TurnResult:
    session_id: str | None
    answer: str
    stats: dict[str, Any]
    trace: dict[str, Any] | None
    error: str | None
    retrieved_skills: list[str]
    heimdall_calls: int
    tool_calls: int
    total_tokens: int
    latency_ms: float | None
    trace_id: str | None
    root_span_id: str | None
    live_snapshot_id: str | None = None
    fingerprint: dict[str, Any] | None = None
