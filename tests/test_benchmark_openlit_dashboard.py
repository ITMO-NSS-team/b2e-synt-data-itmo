"""Automatic OpenLIT benchmark dashboard provisioning."""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping

from sim.benchmark.openlit_dashboard import (
    BOARD_TITLE,
    MODE_COMPARISON_QUERY,
    RUNS_QUERY,
    WIDGETS,
    _render_query,
    provision_dashboard,
)


class FakeStore:
    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = deque(responses)
        self.inserts: list[tuple[str, Mapping[str, Any]]] = []
        self.executions: list[tuple[str, Mapping[str, str]]] = []

    def select(self, query: str, params: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        return self.responses.popleft()

    def execute(self, query: str, params: Mapping[str, str] | None = None) -> None:
        self.executions.append((query, params or {}))

    def insert(self, table: str, row: Mapping[str, Any]) -> None:
        self.inserts.append((table, row))


def _tables() -> list[dict[str, str]]:
    return [
        {"name": "openlit_board"},
        {"name": "openlit_widget"},
        {"name": "openlit_board_widget"},
        {"name": "otel_traces"},
    ]


def test_provision_creates_board_widgets_and_links() -> None:
    store = FakeStore([_tables(), [], [], [], [], [], [], []])

    result = provision_dashboard(store, ui_url="http://openlit.example/")

    assert result.created_board is True
    assert result.created_widgets == 3
    assert [table for table, _ in store.inserts] == [
        "openlit_board",
        "openlit_widget",
        "openlit_board_widget",
        "openlit_widget",
        "openlit_board_widget",
        "openlit_widget",
        "openlit_board_widget",
    ]
    board = store.inserts[0][1]
    assert board["title"] == BOARD_TITLE
    widget_rows = [row for table, row in store.inserts if table == "openlit_widget"]
    assert {row["title"] for row in widget_rows} == {widget.title for widget in WIDGETS}
    assert all("b2e.benchmark" in row["config"] for row in widget_rows)
    runs = next(row for row in widget_rows if row["title"] == "Benchmark runs")
    assert "http://openlit.example" in runs["config"]
    assert "/telemetry/traces/" in runs["config"]


def test_provision_updates_existing_resources_without_duplicates() -> None:
    store = FakeStore([
        _tables(),
        [{"id": "00000000-0000-0000-0000-000000000001"}],
        [{"id": "00000000-0000-0000-0000-000000000002"}],
        [{"id": "00000000-0000-0000-0000-000000000003"}],
        [{"id": "00000000-0000-0000-0000-000000000004"}],
        [{"id": "00000000-0000-0000-0000-000000000005"}],
        [{"id": "00000000-0000-0000-0000-000000000006"}],
        [{"id": "00000000-0000-0000-0000-000000000007"}],
    ])

    result = provision_dashboard(store)

    assert result.created_board is False
    assert result.created_widgets == 0
    assert store.inserts == []
    assert len(store.executions) == 7
    assert all("mutations_sync = 1" in query for query, _ in store.executions)


def test_dashboard_queries_include_trace_links_and_every_mode() -> None:
    rendered = _render_query(RUNS_QUERY, "http://localhost:3000/")

    assert "http://localhost:3000" in rendered
    assert "/telemetry/traces/" in rendered
    assert "SpanId" in rendered
    assert all("/telemetry/traces/" in widget.query for widget in WIDGETS)
    assert "skills_disabled" in MODE_COMPARISON_QUERY
    assert "existing_skills" in MODE_COMPARISON_QUERY
    assert "generated_skills" in MODE_COMPARISON_QUERY
    assert "general_knowledge" in MODE_COMPARISON_QUERY
    assert "accuracy_delta_vs_general_knowledge_pct" in MODE_COMPARISON_QUERY
    assert "accuracy_delta_vs_skills_disabled_pct" in MODE_COMPARISON_QUERY
    assert "accuracy_delta_vs_existing_skills_pct" in MODE_COMPARISON_QUERY
    assert "l.mode != 'generated_skills'" in MODE_COMPARISON_QUERY
    assert "expected_modes" not in MODE_COMPARISON_QUERY


def test_provision_rejects_incomplete_openlit_schema() -> None:
    store = FakeStore([[{"name": "otel_traces"}]])

    try:
        provision_dashboard(store)
    except RuntimeError as exc:
        assert "openlit_board" in str(exc)
    else:  # pragma: no cover - makes the failure message explicit
        raise AssertionError("incomplete schema must fail")
