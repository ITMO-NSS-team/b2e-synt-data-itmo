from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from sim.benchmark.publish import publish_summary


def test_publish_uses_one_final_merged_summary(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest = {
        "eval_id": "eval-1",
        "check_only": False,
        "case_ids": ["case-0001-00", "case-0001-01"],
        "modes": [
            {"name": "existing_skills"},
            {"name": "generated_skills"},
        ],
        "repetitions": 2,
    }
    summary = {"n_runs": 8, "by_mode": {"generated_skills": {"n_runs": 4}}}
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    captured: dict = {}

    class Provider:
        def force_flush(self) -> bool:
            captured["flushed"] = True
            return True

    @contextmanager
    def benchmark_run(**kwargs):
        captured["run"] = kwargs
        yield "span"

    def set_summary(span, value, **kwargs):
        captured["summary"] = (span, value, kwargs)

    monkeypatch.setattr("sim.benchmark.publish.configure_metric_export", lambda: Provider())
    monkeypatch.setattr("sim.benchmark.publish.telemetry.benchmark_run", benchmark_run)
    monkeypatch.setattr("sim.benchmark.publish.telemetry.set_benchmark_summary", set_summary)

    assert publish_summary(tmp_path) is True
    assert captured["run"] == {
        "eval_id": "eval-1",
        "modes": ("existing_skills", "generated_skills"),
        "repetitions": 2,
        "case_count": 2,
        "final": True,
    }
    assert captured["summary"] == (
        "span", summary, {"eval_id": "eval-1", "final": True},
    )
    assert captured["flushed"] is True


def test_publish_is_optional_without_telemetry_endpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    (tmp_path / "run-manifest.json").write_text(json.dumps({
        "eval_id": "eval-1", "check_only": False, "case_ids": [],
        "modes": [], "repetitions": 1,
    }), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({"n_runs": 0}), encoding="utf-8")
    monkeypatch.setattr("sim.benchmark.publish.configure_metric_export", lambda: None)

    assert publish_summary(tmp_path) is False
