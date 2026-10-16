from __future__ import annotations

from types import SimpleNamespace

from sim.benchmark.report import write_report, write_report_from_files


def _result(run_id: str, mode: str, accuracy: int, latency: int, calls: int):
    return SimpleNamespace(
        run_id=run_id, case_id="case-0001-00", mode=mode, repetition=1,
        status="completed", score={"metrics": {
            "answer_accuracy": accuracy,
            "generated_skill_loaded": 1 if "generated" in mode else None,
            "latency_ms": latency,
            "mcp_query_calls": calls,
            "total_tokens": 100 * calls,
        }},
        response={
            "observations": {
                "loaded_skills": ["expected_skill"] if "generated" in mode else [],
            },
            "case_snapshot": {"expected_skills": ["expected_skill"]},
        },
    )


def test_report_has_one_point_per_run_and_aggregate_medians(
    tmp_path, monkeypatch,
) -> None:
    public = tmp_path / "public"
    monkeypatch.setenv("BENCHMARK_REPORT_DIR", str(public))
    results = [
        _result("run-existing", "existing_skills", 0, 1000, 2),
        _result(
            "run-generated",
            "existing_plus_generated@compact--medium--skill-id",
            1, 3000, 4,
        ),
    ]

    text = write_report(tmp_path / "eval-1", results).read_text(encoding="utf-8")

    assert "run-existing" in text
    assert "run-generated" in text
    assert "Время × вызовы mcp_query" in text
    assert "Answer_accuracy × токены" in text
    assert "Число вызовов mcp_query, шт." in text
    assert "Время выполнения, с" in text
    assert "Медиана total_tokens, токены" in text
    assert "Answer_accuracy, %" in text
    assert "Загрузка skills по конфигурациям" not in text
    assert "размер точки" not in text
    assert 'id="efficiency" class="plot"' in text
    assert 'id="quality-cost" class="plot"' in text
    assert "Plotly.newPlot" in text
    assert "scrollZoom: true" in text
    assert "efficiency-zoom" not in text
    assert "new ResizeObserver(drawAxes)" not in text
    assert "Развернуть" in text
    assert "Нажмите на точку" in text
    assert "Любой skill загружен" in text
    assert "Expected skill загружен" in text
    assert '"method":"compact"' in text
    assert '"detail":"medium"' in text
    assert '"median_latency_s":3.0' in text
    assert '"median_total_tokens":400.0' in text
    assert '"expected_skill_loaded":1.0' in text
    assert '"delta_pp":100.0' in text
    assert (public / "eval-1" / "report.html").read_text() == text
    assert list((public / "eval-1").iterdir()) == [public / "eval-1" / "report.html"]


def test_old_result_files_backfill_skill_metrics(tmp_path) -> None:
    root = tmp_path / "old-eval"
    root.mkdir()
    (root / "scores.jsonl").write_text(
        '{"run_id":"run-1","metrics":{"answer_accuracy":1}}\n',
        encoding="utf-8",
    )
    (root / "responses.jsonl").write_text(
        '{"run_id":"run-1","case_id":"case-1","mode":"existing_skills",'
        '"repetition":1,"status":"completed","observations":'
        '{"loaded_skills":["expected"]},"case_snapshot":'
        '{"expected_skills":["expected"]}}\n',
        encoding="utf-8",
    )

    text = write_report_from_files(root).read_text(encoding="utf-8")

    assert '"any_skill_loaded":1' in text
    assert '"expected_skill_loaded":1' in text
