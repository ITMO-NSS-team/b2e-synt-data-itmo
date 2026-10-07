"""Idempotently provision the benchmark dashboard in OpenLIT.

OpenLIT stores custom dashboards in the same ClickHouse database as its OTLP
tables.  Provisioning the small, versioned layout directly keeps benchmark
runs non-interactive: no browser login or one-off import is required on a new
stand.  The operation is opt-in through ``OPENLIT_CLICKHOUSE_URL`` and never
affects benchmark execution when OpenLIT is not configured.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import httpx

from .env import load_env


BOARD_TITLE = "B2E Benchmark"
BOARD_DESCRIPTION = (
    "Запуски эталонного бенчмарка и агрегированные метрики по режимам."
)
BOARD_TAGS = ["benchmark", "managed:b2e"]

_NAMESPACE = uuid.UUID("c6428d8e-38ce-4f8d-aa85-9dc0b692b06e")
DEFAULT_BOARD_ID = str(uuid.uuid5(_NAMESPACE, "dashboard"))

RUNS_QUERY = """
WITH
    parseDateTimeBestEffort('{{filter.timeLimit.start}}') AS start_time,
    parseDateTimeBestEffort('{{filter.timeLimit.end}}') AS end_time
SELECT
    formatDateTime(Timestamp, '%Y-%m-%d %H:%i:%S') AS started_at,
    SpanAttributes['b2e.benchmark.eval_id'] AS eval_id,
    SpanAttributes['b2e.benchmark.modes'] AS modes,
    toUInt32OrZero(SpanAttributes['b2e.benchmark.case_count']) AS cases,
    toUInt32OrZero(SpanAttributes['b2e.summary.n_runs']) AS runs,
    toUInt32OrZero(SpanAttributes['b2e.summary.n_completed']) AS completed,
    toUInt32OrZero(SpanAttributes['b2e.summary.n_scored']) AS scored,
    toUInt32OrZero(SpanAttributes['b2e.summary.n_unscored']) AS unscored,
    round(Duration / 1000000000.0, 1) AS duration_s,
    concat(__OPENLIT_UI_URL__, '/telemetry/traces/', SpanId) AS trace_url
FROM otel_traces
WHERE
    ServiceName = 'b2e-benchmark-runner'
    AND SpanName = 'b2e.benchmark.run'
    AND SpanAttributes['b2e.benchmark.final'] != 'false'
    AND Timestamp >= start_time AND Timestamp <= end_time
ORDER BY Timestamp DESC
LIMIT 100
""".strip()

MODE_SUMMARY_QUERY = """
WITH
    parseDateTimeBestEffort('{{filter.timeLimit.start}}') AS start_time,
    parseDateTimeBestEffort('{{filter.timeLimit.end}}') AS end_time
SELECT
    formatDateTime(Timestamp, '%Y-%m-%d %H:%i:%S') AS finished_at,
    SpanAttributes['b2e.benchmark.eval_id'] AS eval_id,
    SpanAttributes['b2e.benchmark.mode'] AS mode,
    toUInt32OrZero(SpanAttributes['b2e.summary.n_runs']) AS runs,
    round(toFloat64OrNull(nullIf(
        SpanAttributes['b2e.summary.answer_accuracy'], '')) * 100, 1)
        AS answer_accuracy_pct,
    round(toFloat64OrNull(nullIf(
        SpanAttributes['b2e.summary.exact_match'], '')) * 100, 1)
        AS exact_match_pct,
    round(toFloat64OrNull(nullIf(
        SpanAttributes['b2e.summary.correct_refusal'], '')) * 100, 1)
        AS refusal_pct,
    round(toFloat64OrNull(nullIf(
        SpanAttributes['b2e.summary.mcp_query_calls'], '')), 1)
        AS mcp_queries,
    round(toFloat64OrNull(nullIf(
        SpanAttributes['b2e.summary.failed_tool_calls'], '')), 1)
        AS failed_calls,
    round(toFloat64OrNull(nullIf(
        SpanAttributes['b2e.summary.total_tokens'], '')), 0)
        AS tokens,
    round(toFloat64OrNull(nullIf(
        SpanAttributes['b2e.summary.latency_ms'], '')) / 1000, 1)
        AS latency_s,
    concat(__OPENLIT_UI_URL__, '/telemetry/traces/', SpanId) AS trace_url
FROM otel_traces
WHERE
    ServiceName = 'b2e-benchmark-runner'
    AND startsWith(SpanName, 'b2e.benchmark.summary.')
    AND SpanAttributes['b2e.benchmark.final'] != 'false'
    AND Timestamp >= start_time AND Timestamp <= end_time
ORDER BY Timestamp DESC, eval_id, mode
LIMIT 300
""".strip()

MODE_COMPARISON_QUERY = """
WITH
    parseDateTimeBestEffort('{{filter.timeLimit.start}}') AS start_time,
    parseDateTimeBestEffort('{{filter.timeLimit.end}}') AS end_time,
    summaries AS (
        SELECT
            Timestamp,
            SpanId AS span_id,
            SpanAttributes['b2e.benchmark.eval_id'] AS eval_id,
            SpanAttributes['b2e.benchmark.mode'] AS mode,
            toUInt32OrZero(SpanAttributes['b2e.summary.n_runs']) AS runs,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.summary.answer_accuracy'], '')) AS accuracy,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.summary.mcp_query_calls'], '')) AS mcp_queries,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.summary.total_tokens'], '')) AS tokens,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.summary.latency_ms'], '')) / 1000 AS latency_s
        FROM otel_traces
        WHERE
            ServiceName = 'b2e-benchmark-runner'
            AND startsWith(SpanName, 'b2e.benchmark.summary.')
            AND SpanAttributes['b2e.benchmark.final'] != 'false'
            AND Timestamp >= start_time AND Timestamp <= end_time
    ),
    latest AS (
        SELECT
            eval_id,
            mode,
            max(Timestamp) AS finished_at,
            argMax(runs, Timestamp) AS runs,
            argMax(accuracy, Timestamp) AS accuracy,
            argMax(mcp_queries, Timestamp) AS mcp_queries,
            argMax(tokens, Timestamp) AS tokens,
            argMax(latency_s, Timestamp) AS latency_s,
            argMax(span_id, Timestamp) AS span_id
        FROM summaries
        GROUP BY eval_id, mode
    ),
    baseline AS (
        SELECT
            eval_id,
            countIf(mode = 'general_knowledge') AS general_present,
            anyIf(accuracy, mode = 'general_knowledge') AS general_accuracy,
            countIf(mode = 'skills_disabled') AS disabled_present,
            anyIf(accuracy, mode = 'skills_disabled') AS disabled_accuracy,
            countIf(mode = 'existing_skills') AS existing_present,
            anyIf(accuracy, mode = 'existing_skills') AS existing_accuracy
        FROM latest
        GROUP BY eval_id
    )
SELECT
    formatDateTime(l.finished_at, '%Y-%m-%d %H:%i:%S') AS finished_at,
    l.eval_id AS eval_id,
    l.mode AS mode,
    l.runs AS runs,
    round(l.accuracy * 100, 1) AS answer_accuracy_pct,
    if(
        l.mode != 'generated_skills' AND b.general_present > 0,
        round((l.accuracy - b.general_accuracy) * 100, 1),
        NULL
    ) AS accuracy_delta_vs_general_knowledge_pct,
    if(
        l.mode != 'generated_skills' AND b.disabled_present > 0,
        round((l.accuracy - b.disabled_accuracy) * 100, 1),
        NULL
    ) AS accuracy_delta_vs_skills_disabled_pct,
    if(
        l.mode != 'generated_skills' AND b.existing_present > 0,
        round((l.accuracy - b.existing_accuracy) * 100, 1),
        NULL
    ) AS accuracy_delta_vs_existing_skills_pct,
    round(l.mcp_queries, 1) AS mcp_queries,
    round(l.tokens, 0) AS tokens,
    round(l.latency_s, 1) AS latency_s,
    concat(__OPENLIT_UI_URL__, '/telemetry/traces/', l.span_id) AS trace_url
FROM latest AS l
LEFT JOIN baseline AS b ON b.eval_id = l.eval_id
ORDER BY l.finished_at DESC, l.eval_id, indexOf([
    'general_knowledge',
    'skills_disabled',
    'existing_skills',
    'generated_skills'
], splitByChar('@', l.mode)[1])
LIMIT 400
""".strip()


@dataclass(frozen=True)
class WidgetSpec:
    key: str
    title: str
    description: str
    query: str
    position: Mapping[str, int]

    @property
    def id(self) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"widget:{self.key}"))


WIDGETS = (
    WidgetSpec(
        key="runs",
        title="Benchmark runs",
        description="Один завершённый запуск benchmark на строку.",
        query=RUNS_QUERY,
        position={"x": 0, "y": 0, "w": 4, "h": 2},
    ),
    WidgetSpec(
        key="mode-summary",
        title="Metrics by mode",
        description=(
            "Агрегированные метрики режима. answer_accuracy_pct — доля "
            "правильно решённых среди оценённых задач, в процентах."
        ),
        query=MODE_SUMMARY_QUERY,
        position={"x": 0, "y": 2, "w": 4, "h": 3},
    ),
    WidgetSpec(
        key="mode-comparison",
        title="Mode comparison",
        description=(
            "Сопоставление режимов по доле правильно решённых задач и "
            "дельтам в процентных пунктах относительно general_knowledge, "
            "skills_disabled и existing_skills. Строки создаются только для "
            "запущенных режимов; каждая generated-вариация показана отдельно."
        ),
        query=MODE_COMPARISON_QUERY,
        position={"x": 0, "y": 5, "w": 4, "h": 3},
    ),
)


@dataclass(frozen=True)
class DashboardConfig:
    url: str
    username: str = "default"
    password: str = ""
    database: str = "openlit"
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> "DashboardConfig | None":
        url = (os.getenv("OPENLIT_CLICKHOUSE_URL") or "").strip()
        if not url:
            return None
        return cls(
            url=url.rstrip("/"),
            username=os.getenv("OPENLIT_DB_USER", "default"),
            password=os.getenv("OPENLIT_DB_PASSWORD", ""),
            database=os.getenv("OPENLIT_DB_NAME", "openlit"),
            timeout=float(os.getenv("OPENLIT_DASHBOARD_TIMEOUT", "15")),
        )


class DashboardStore(Protocol):
    def select(self, query: str, params: Mapping[str, str] | None = None) -> list[dict[str, Any]]: ...
    def execute(self, query: str, params: Mapping[str, str] | None = None) -> None: ...
    def insert(self, table: str, row: Mapping[str, Any]) -> None: ...


class ClickHouseHTTPStore:
    """Minimal ClickHouse HTTP client used only by the provisioner."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            base_url=config.url,
            auth=(config.username, config.password),
            timeout=config.timeout,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def _post(self, query: str, params: Mapping[str, str] | None = None) -> str:
        query_params: dict[str, str] = {"database": self.config.database}
        query_params.update({f"param_{key}": value for key, value in (params or {}).items()})
        response = self._client.post("/", params=query_params, content=query.encode("utf-8"))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            raise RuntimeError(
                f"ClickHouse returned HTTP {response.status_code}: {detail}"
            ) from exc
        return response.text

    def select(self, query: str, params: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        payload = self._post(f"{query.rstrip().rstrip(';')} FORMAT JSONEachRow", params)
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    def execute(self, query: str, params: Mapping[str, str] | None = None) -> None:
        self._post(query, params)

    def insert(self, table: str, row: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
        self._post(f"INSERT INTO {table} FORMAT JSONEachRow\n{payload}")


@dataclass(frozen=True)
class ProvisionResult:
    board_id: str
    created_board: bool
    created_widgets: int
    dashboard_url: str | None = None


def provision_dashboard(
    store: DashboardStore, *, ui_url: str = "",
) -> ProvisionResult:
    """Create or refresh the managed board and its widgets."""
    _require_tables(store)
    board_rows = store.select(
        "SELECT toString(id) AS id FROM openlit_board "
        "WHERE title = {title:String} ORDER BY updated_at DESC LIMIT 1",
        {"title": BOARD_TITLE},
    )
    created_board = not board_rows
    board_id = board_rows[0]["id"] if board_rows else DEFAULT_BOARD_ID
    board_row = {
        "id": board_id,
        "title": BOARD_TITLE,
        "description": BOARD_DESCRIPTION,
        "parent_id": None,
        "is_main_dashboard": False,
        "is_pinned": True,
        "tags": json.dumps(BOARD_TAGS, ensure_ascii=False),
    }
    if created_board:
        store.insert("openlit_board", board_row)
    else:
        _update_board(store, board_row)

    created_widgets = 0
    for widget in WIDGETS:
        widget_rows = store.select(
            "SELECT toString(id) AS id FROM openlit_widget "
            # Qualify the source columns: ClickHouse otherwise substitutes the
            # String result alias `id` into WHERE and compares String to UUID.
            "WHERE openlit_widget.id = toUUID({id:String}) "
            "OR openlit_widget.title = {title:String} "
            "ORDER BY updated_at DESC LIMIT 1",
            {"id": widget.id, "title": widget.title},
        )
        widget_id = widget_rows[0]["id"] if widget_rows else widget.id
        widget_row = {
            "id": widget_id,
            "title": widget.title,
            "description": widget.description,
            "widget_type": "TABLE",
            "properties": json.dumps(
                {"color": "#F36C06", "autoRefresh": True},
                separators=(",", ":"),
            ),
            "config": json.dumps(
                {"query": _render_query(widget.query, ui_url)},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        if widget_rows:
            _update_widget(store, widget_row)
        else:
            store.insert("openlit_widget", widget_row)
            created_widgets += 1

        mapping_rows = store.select(
            "SELECT toString(id) AS id FROM openlit_board_widget "
            "WHERE board_id = toUUID({board_id:String}) "
            "AND widget_id = toUUID({widget_id:String}) LIMIT 1",
            {"board_id": board_id, "widget_id": widget_id},
        )
        position = json.dumps(widget.position, separators=(",", ":"))
        if mapping_rows:
            store.execute(
                "ALTER TABLE openlit_board_widget UPDATE "
                "position = {position:String}, updated_at = now() "
                "WHERE id = toUUID({id:String}) SETTINGS mutations_sync = 1",
                {"position": position, "id": mapping_rows[0]["id"]},
            )
        else:
            store.insert(
                "openlit_board_widget",
                {
                    "id": str(uuid.uuid5(_NAMESPACE, f"mapping:{board_id}:{widget_id}")),
                    "board_id": board_id,
                    "widget_id": widget_id,
                    "position": position,
                },
            )

    return ProvisionResult(
        board_id=board_id,
        created_board=created_board,
        created_widgets=created_widgets,
    )


def _render_query(query: str, ui_url: str) -> str:
    """Insert the deployment-specific UI base into stored trace links."""
    base = ui_url.rstrip("/")
    literal = "'" + base.replace("'", "''") + "'"
    return query.replace("__OPENLIT_UI_URL__", literal)


def _require_tables(store: DashboardStore) -> None:
    expected = {"openlit_board", "openlit_widget", "openlit_board_widget", "otel_traces"}
    rows = store.select(
        "SELECT name FROM system.tables WHERE database = currentDatabase() "
        "AND name IN ({boards:String}, {widgets:String}, {links:String}, {traces:String})",
        {
            "boards": "openlit_board",
            "widgets": "openlit_widget",
            "links": "openlit_board_widget",
            "traces": "otel_traces",
        },
    )
    missing = expected.difference(str(row.get("name")) for row in rows)
    if missing:
        raise RuntimeError(
            "OpenLIT ClickHouse is missing required tables: " + ", ".join(sorted(missing))
        )


def _update_board(store: DashboardStore, row: Mapping[str, Any]) -> None:
    store.execute(
        "ALTER TABLE openlit_board UPDATE "
        "title = {title:String}, description = {description:String}, "
        "is_pinned = true, tags = {tags:String}, updated_at = now() "
        "WHERE id = toUUID({id:String}) SETTINGS mutations_sync = 1",
        {key: str(row[key]) for key in ("id", "title", "description", "tags")},
    )


def _update_widget(store: DashboardStore, row: Mapping[str, Any]) -> None:
    store.execute(
        "ALTER TABLE openlit_widget UPDATE "
        "title = {title:String}, description = {description:String}, "
        "widget_type = {widget_type:String}, properties = {properties:String}, "
        "config = {config:String}, updated_at = now() "
        "WHERE id = toUUID({id:String}) SETTINGS mutations_sync = 1",
        {key: str(row[key]) for key in (
            "id", "title", "description", "widget_type", "properties", "config",
        )},
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Provision the B2E benchmark dashboard in OpenLIT")
    result.add_argument("--env-file", default="deploy/.env")
    result.add_argument("--required", action="store_true", help="fail if OpenLIT dashboard access is not configured")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_env(args.env_file)
    config = DashboardConfig.from_env()
    if config is None:
        message = "OPENLIT_CLICKHOUSE_URL is empty; dashboard provisioning skipped"
        if args.required:
            raise SystemExit(message)
        print(message)
        return 0

    store = ClickHouseHTTPStore(config)
    try:
        dashboard_url = (os.getenv("OPENLIT_UI_URL") or "").rstrip("/")
        result = provision_dashboard(store, ui_url=dashboard_url)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        if args.required:
            raise SystemExit(f"OpenLIT dashboard provisioning failed: {exc}") from exc
        print(f"OpenLIT dashboard provisioning skipped: {exc}")
        return 0
    finally:
        store.close()

    suffix = f"{dashboard_url}/d/{result.board_id}" if dashboard_url else result.board_id
    print(
        "OpenLIT benchmark dashboard ready: "
        f"{suffix} (created_board={result.created_board}, "
        f"created_widgets={result.created_widgets})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
