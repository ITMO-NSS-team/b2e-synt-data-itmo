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
from dataclasses import dataclass, field
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
    toUInt32OrZero(SpanAttributes['b2e.benchmark.case_count']) AS cases,
    toUInt32OrZero(SpanAttributes['b2e.summary.n_runs']) AS runs,
    toUInt32OrZero(SpanAttributes['b2e.summary.n_scored']) AS scored,
    concat(
        __BENCHMARK_REPORTS_URL__, '/',
        SpanAttributes['b2e.benchmark.eval_id'], '/report.html'
    ) AS report_url,
    concat(__OPENLIT_UI_URL__, '/telemetry/traces/', SpanId) AS trace_url
FROM otel_traces
WHERE
    ServiceName = 'b2e-benchmark-runner'
    AND SpanName = 'b2e.benchmark.run'
    AND SpanAttributes['b2e.benchmark.final'] != 'false'
    AND Timestamp >= start_time AND Timestamp <= end_time
ORDER BY Timestamp DESC
LIMIT 50
""".strip()

_LATEST_SCORE_CTES = """
WITH
    parseDateTimeBestEffort('{{filter.timeLimit.start}}') AS start_time,
    parseDateTimeBestEffort('{{filter.timeLimit.end}}') AS end_time,
    latest_eval AS (
        SELECT
            argMax(SpanAttributes['b2e.benchmark.eval_id'], Timestamp) AS eval_id
        FROM otel_traces
        WHERE
            ServiceName = 'b2e-benchmark-runner'
            AND SpanName = 'b2e.benchmark.run'
            AND SpanAttributes['b2e.benchmark.final'] != 'false'
            AND Timestamp >= start_time AND Timestamp <= end_time
    ),
    scores AS (
        SELECT
            SpanAttributes['b2e.benchmark.eval_id'] AS eval_id,
            SpanAttributes['b2e.benchmark.case_id'] AS case_id,
            toUInt32OrZero(SpanAttributes['b2e.benchmark.repetition']) AS repetition,
            splitByChar('@', SpanAttributes['b2e.benchmark.mode'])[1] AS mode,
            splitByChar('@', SpanAttributes['b2e.benchmark.mode'])[2] AS variant,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.metric.answer_accuracy'], '')) AS answer_score,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.metric.generated_skill_loaded'], '')) AS skill_loaded_score,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.metric.latency_ms'], '')) AS latency_ms,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.metric.mcp_query_calls'], '')) AS mcp_queries,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.metric.prompt_tokens'], '')) AS input_tokens,
            toFloat64OrNull(nullIf(
                SpanAttributes['b2e.metric.completion_tokens'], '')) AS output_tokens
        FROM otel_traces
        WHERE
            ServiceName = 'b2e-benchmark-runner'
            AND SpanAttributes['b2e.benchmark.span_type'] = 'score'
            AND SpanAttributes['b2e.benchmark.eval_id'] = (
                SELECT eval_id FROM latest_eval
            )
            AND Timestamp >= start_time AND Timestamp <= end_time
    ),
    grouped AS (
        SELECT
            eval_id,
            mode,
            if(variant = '', '—', variant) AS configuration,
            avg(answer_score) AS answer_accuracy,
            count(answer_score) AS answer_scored,
            countIf(answer_score = 1) AS answer_correct,
            avg(skill_loaded_score) AS target_skill_loaded,
            count(skill_loaded_score) AS target_eligible,
            countIf(skill_loaded_score = 1) AS target_loaded,
            quantileExactInclusive(0.5)(latency_ms) / 1000 AS median_latency_s,
            quantileExactInclusive(0.5)(mcp_queries) AS median_mcp_queries,
            quantileExactInclusive(0.5)(input_tokens) AS median_input_tokens,
            quantileExactInclusive(0.5)(output_tokens) AS median_output_tokens
        FROM scores
        GROUP BY eval_id, mode, configuration
    ),
    baseline AS (
        SELECT
            eval_id,
            countIf(mode = 'existing_skills') AS existing_present
        FROM grouped
        GROUP BY eval_id
    ),
    paired AS (
        SELECT
            generated.eval_id AS eval_id,
            generated.mode AS mode,
            generated.variant AS variant,
            avg(generated.answer_score - existing.answer_score) * 100 AS delta_pp
        FROM scores AS generated
        INNER JOIN scores AS existing
            ON generated.eval_id = existing.eval_id
            AND generated.case_id = existing.case_id
            AND generated.repetition = existing.repetition
        WHERE existing.mode = 'existing_skills'
            AND generated.mode != 'existing_skills'
            AND generated.answer_score IS NOT NULL
            AND existing.answer_score IS NOT NULL
        GROUP BY generated.eval_id, generated.mode, generated.variant
    )
""".strip()

MODE_COMPARISON_QUERY = _LATEST_SCORE_CTES + "\n" + """
SELECT
    g.mode AS mode,
    g.configuration AS configuration,
    if(g.answer_scored = 0, NULL, concat(
        toString(round(g.answer_accuracy * 100, 1)), '% (',
        toString(g.answer_correct), '/', toString(g.answer_scored), ')'
    )) AS answer_accuracy_pct,
    if(
        g.mode = 'existing_skills' OR b.existing_present = 0,
        NULL,
        round(p.delta_pp, 1)
    ) AS accuracy_delta_vs_existing_skills_pp,
    if(
        g.mode IN ('generated_skills', 'existing_plus_generated', 'existing_plus_catalog')
            AND g.target_eligible > 0,
        concat(toString(round(g.target_skill_loaded * 100, 1)), '% (',
            toString(g.target_loaded), '/', toString(g.target_eligible), ')'),
        NULL
    ) AS target_skill_loaded_pct,
    round(g.median_latency_s, 1) AS median_latency_s,
    round(g.median_mcp_queries, 1) AS median_mcp_queries,
    round(g.median_input_tokens, 1) AS median_input_tokens,
    round(g.median_output_tokens, 1) AS median_output_tokens
FROM grouped AS g
LEFT JOIN baseline AS b ON b.eval_id = g.eval_id
LEFT JOIN paired AS p ON p.eval_id = g.eval_id
    AND p.mode = g.mode AND if(p.variant = '', '—', p.variant) = g.configuration
ORDER BY indexOf([
    'general_knowledge',
    'skills_disabled',
    'existing_skills',
    'generated_skills',
    'existing_plus_generated',
    'existing_plus_catalog'
], g.mode), g.configuration
""".strip()

ACCURACY_CHART_QUERY = _LATEST_SCORE_CTES + "\n" + """
SELECT
    result.1 AS answer_result,
    result.2 AS share_pct
FROM grouped
ARRAY JOIN [
    tuple(
        concat(if(configuration = '—', mode, concat(mode, ' / ', configuration)), ' · correct'),
        round(answer_accuracy * 100, 1)
    ),
    tuple(
        concat(if(configuration = '—', mode, concat(mode, ' / ', configuration)), ' · incorrect'),
        round((1 - answer_accuracy) * 100, 1)
    )
] AS result
WHERE answer_accuracy IS NOT NULL
ORDER BY answer_result
""".strip()

LATENCY_CHART_QUERY = _LATEST_SCORE_CTES + "\n" + """
SELECT
    if(g.configuration = '—', g.mode, concat(g.mode, ' / ', g.configuration)) AS configuration,
    round(g.median_latency_s, 1) AS median_latency_s
FROM grouped AS g
WHERE g.median_latency_s IS NOT NULL
ORDER BY median_latency_s DESC, configuration
""".strip()

# Kept as a public alias for callers that imported the previous query name.
MODE_SUMMARY_QUERY = LATENCY_CHART_QUERY

SKILL_LOADING_CHART_QUERY = _LATEST_SCORE_CTES + "\n" + """
SELECT
    concat(g.mode, ' / ', g.configuration) AS configuration,
    round(g.target_skill_loaded * 100, 1) AS target_skill_loaded_pct
FROM grouped AS g
WHERE
    g.mode IN ('generated_skills', 'existing_plus_generated', 'existing_plus_catalog')
    AND g.target_skill_loaded IS NOT NULL
ORDER BY target_skill_loaded_pct DESC, configuration
""".strip()


@dataclass(frozen=True)
class WidgetSpec:
    key: str
    title: str
    description: str
    query: str
    position: Mapping[str, int]
    widget_type: str = "TABLE"
    properties: Mapping[str, Any] = field(default_factory=dict)

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
        title="Median execution time — latest experiment",
        description=(
            "Медианное время решения в секундах; устойчиво к единичным "
            "длинным запускам."
        ),
        query=MODE_SUMMARY_QUERY,
        position={"x": 0, "y": 2, "w": 2, "h": 2},
        widget_type="BAR_CHART",
        properties={"xAxis": "configuration", "yAxis": "median_latency_s"},
    ),
    WidgetSpec(
        key="mode-comparison",
        title="Latest experiment metrics",
        description=(
            "Последний benchmark в выбранном интервале: accuracy, прирост к "
            "existing_skills, загрузка целевого generated skill и медианы "
            "времени и mcp_query. Вариант показывается целиком; прирост считается "
            "только по одинаковым кейсам и повторам."
        ),
        query=MODE_COMPARISON_QUERY,
        position={"x": 0, "y": 4, "w": 4, "h": 3},
    ),
    WidgetSpec(
        key="accuracy-chart",
        title="Correct / incorrect answers — latest experiment",
        description=(
            "Доли правильных и неправильных ответов. В отличие от обычного "
            "accuracy bar, график остаётся видимым при результате 0%."
        ),
        query=ACCURACY_CHART_QUERY,
        position={"x": 2, "y": 2, "w": 2, "h": 2},
        widget_type="BAR_CHART",
        properties={"xAxis": "answer_result", "yAxis": "share_pct"},
    ),
    WidgetSpec(
        key="skill-loading-chart",
        title="Target skill loading — latest experiment",
        description=(
            "Доля запусков, в которых агент загрузил соответствующий "
            "generated skill."
        ),
        query=SKILL_LOADING_CHART_QUERY,
        position={"x": 0, "y": 7, "w": 4, "h": 2},
        widget_type="BAR_CHART",
        properties={"xAxis": "configuration", "yAxis": "target_skill_loaded_pct"},
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
    store: DashboardStore, *, ui_url: str = "", reports_url: str = "",
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
            "widget_type": widget.widget_type,
            "properties": json.dumps(
                {"color": "#F36C06", "autoRefresh": True, **widget.properties},
                separators=(",", ":"),
            ),
            "config": json.dumps(
                {"query": _render_query(widget.query, ui_url, reports_url)},
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


def _render_query(query: str, ui_url: str, reports_url: str = "") -> str:
    """Insert the deployment-specific UI base into stored trace links."""
    base = ui_url.rstrip("/")
    report_base = reports_url.rstrip("/")
    literal = "'" + base.replace("'", "''") + "'"
    report_literal = "'" + report_base.replace("'", "''") + "'"
    return query.replace("__OPENLIT_UI_URL__", literal).replace(
        "__BENCHMARK_REPORTS_URL__", report_literal,
    )


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
        "config = unhex({config_hex:String}), updated_at = now() "
        "WHERE id = toUUID({id:String}) SETTINGS mutations_sync = 1",
        {
            **{key: str(row[key]) for key in (
                "id", "title", "description", "widget_type", "properties",
            )},
            "config_hex": str(row["config"]).encode("utf-8").hex(),
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Provision the B2E benchmark dashboard in OpenLIT")
    result.add_argument("--env-file", default="deploy/.env")
    result.add_argument("--required", action="store_true", help="fail if OpenLIT dashboard access is not configured")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_env(args.env_file, override=True)
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
        public_url = (os.getenv("PUBLIC_URL") or "").rstrip("/")
        reports_url = (
            os.getenv("BENCHMARK_REPORTS_URL")
            or (f"{public_url}/benchmark-reports" if public_url else "")
        ).rstrip("/")
        result = provision_dashboard(
            store, ui_url=dashboard_url, reports_url=reports_url,
        )
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
