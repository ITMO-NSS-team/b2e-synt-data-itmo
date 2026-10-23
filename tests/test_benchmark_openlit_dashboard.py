"""Automatic OpenLIT benchmark dashboard provisioning."""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping

from sim.benchmark.openlit_dashboard import (
    ACCURACY_CHART_QUERY,
    BOARD_TITLE,
    LATENCY_CHART_QUERY,
    MODE_COMPARISON_QUERY,
    MODE_SUMMARY_QUERY,
    RUNS_QUERY,
    SKILL_LOADING_CHART_QUERY,
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
    store = FakeStore([_tables(), [], *([[]] * (len(WIDGETS) * 2))])

    result = provision_dashboard(
        store,
        ui_url="http://openlit.example/",
        reports_url="https://stand.example/benchmark-reports/",
    )

    assert result.created_board is True
    assert result.created_widgets == len(WIDGETS)
    assert [table for table, _ in store.inserts] == [
        "openlit_board",
        *(["openlit_widget", "openlit_board_widget"] * len(WIDGETS)),
    ]
    board = store.inserts[0][1]
    assert board["title"] == BOARD_TITLE
    widget_rows = [row for table, row in store.inserts if table == "openlit_widget"]
    assert {row["title"] for row in widget_rows} == {widget.title for widget in WIDGETS}
    assert all("b2e.benchmark" in row["config"] for row in widget_rows)
    runs = next(row for row in widget_rows if row["title"] == "Benchmark runs")
    assert "http://openlit.example" in runs["config"]
    assert "/telemetry/traces/" in runs["config"]
    assert "https://stand.example/benchmark-reports" in runs["config"]
    charts = [row for row in widget_rows if row["widget_type"] == "BAR_CHART"]
    assert len(charts) == 3
    assert all('"xAxis"' in row["properties"] for row in charts)
    assert all('"yAxis"' in row["properties"] for row in charts)


def test_provision_updates_existing_resources_without_duplicates() -> None:
    existing = [
        [{"id": f"00000000-0000-0000-0000-{index:012d}"}]
        for index in range(1, 2 + len(WIDGETS) * 2)
    ]
    store = FakeStore([_tables(), *existing])

    result = provision_dashboard(store)

    assert result.created_board is False
    assert result.created_widgets == 0
    assert store.inserts == []
    assert len(store.executions) == 1 + len(WIDGETS) * 2
    assert all("mutations_sync = 1" in query for query, _ in store.executions)


def test_dashboard_queries_include_trace_links_and_every_mode() -> None:
    rendered = _render_query(
        RUNS_QUERY,
        "http://localhost:3000/",
        "https://localhost:8443/benchmark-reports/",
    )

    assert "http://localhost:3000" in rendered
    assert "/telemetry/traces/" in rendered
    assert "https://localhost:8443/benchmark-reports" in rendered
    assert "SpanId" in rendered
    assert "skills_disabled" in MODE_COMPARISON_QUERY
    assert "existing_skills" in MODE_COMPARISON_QUERY
    assert "generated_skills" in MODE_COMPARISON_QUERY
    assert "general_knowledge" in MODE_COMPARISON_QUERY
    assert "accuracy_delta_vs_existing_skills_pp" in MODE_COMPARISON_QUERY
    assert "('generated_skills', 'existing_plus_generated')" in MODE_COMPARISON_QUERY
    assert "quantileExactInclusive(0.5)(latency_ms)" in MODE_COMPARISON_QUERY
    assert "quantileExactInclusive(0.5)(mcp_queries)" in MODE_COMPARISON_QUERY
    assert "latest_eval" in MODE_COMPARISON_QUERY
    assert "g.configuration AS configuration" in MODE_COMPARISON_QUERY
    assert "generated.case_id = existing.case_id" in MODE_COMPARISON_QUERY
    assert "generated.repetition = existing.repetition" in MODE_COMPARISON_QUERY
    assert "AS method" not in MODE_COMPARISON_QUERY
    assert "share_pct" in ACCURACY_CHART_QUERY
    assert "correct" in ACCURACY_CHART_QUERY
    assert "incorrect" in ACCURACY_CHART_QUERY
    assert "median_latency_s" in LATENCY_CHART_QUERY
    assert "target_skill_loaded_pct" in SKILL_LOADING_CHART_QUERY
    assert all(
        "b2e.benchmark.final" in query
        for query in (
            RUNS_QUERY,
            MODE_SUMMARY_QUERY,
            MODE_COMPARISON_QUERY,
            ACCURACY_CHART_QUERY,
            LATENCY_CHART_QUERY,
            SKILL_LOADING_CHART_QUERY,
        )
    )


def test_provision_rejects_incomplete_openlit_schema() -> None:
    store = FakeStore([[{"name": "otel_traces"}]])

    try:
        provision_dashboard(store)
    except RuntimeError as exc:
        assert "openlit_board" in str(exc)
    else:  # pragma: no cover - makes the failure message explicit
        raise AssertionError("incomplete schema must fail")
