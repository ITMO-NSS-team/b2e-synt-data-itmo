from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.benchmark.merge import merge_phases


def _phase(root: Path, name: str, mode: str, *, check_only: bool = False) -> Path:
    phase = root / name
    phase.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0",
        "eval_id": "eval-1",
        "check_only": check_only,
        "cases_path": "/cases",
        "case_ids": ["case-0001-00"],
        "modes": [{"name": mode}],
        "repetitions": 0 if check_only else 1,
        "live_stand": {"phase": name},
    }
    (phase / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return phase


def _run(phase: Path, mode: str, *, with_trace: bool = True) -> None:
    run_id = f"eval-1-case-0001-00-{mode}-01"
    response = {
        "run_id": run_id,
        "comparison_group_id": "eval-1-case-0001-00-01",
        "case_id": "case-0001-00",
        "mode": mode,
        "repetition": 1,
        "status": "completed",
        "response": {"raw_answer": "{}", "error": None},
    }
    if with_trace:
        (phase / "traces").mkdir()
        trace_path = f"traces/{run_id}.json"
        (phase / trace_path).write_text(json.dumps({"spans": [{"name": mode}]}), encoding="utf-8")
        response["trace_path"] = trace_path
    (phase / "responses.jsonl").write_text(json.dumps(response) + "\n", encoding="utf-8")
    score = {"run_id": run_id, "metrics": {"answer_accuracy": 1}, "reasons": []}
    (phase / "scores.jsonl").write_text(json.dumps(score) + "\n", encoding="utf-8")


def test_merge_phases_writes_one_summary_and_keeps_shared_group(tmp_path: Path) -> None:
    base = _phase(tmp_path, "base", "existing_skills")
    generated = _phase(tmp_path, "generated", "generated_skills")
    _run(base, "existing_skills")
    _run(generated, "generated_skills")

    results, writer, check_only = merge_phases(
        [base, generated], results_root=tmp_path / "results", eval_id="eval-1",
    )

    assert check_only is False
    assert len(results) == 2
    assert {row.comparison_group_id for row in results} == {"eval-1-case-0001-00-01"}
    summary = json.loads((writer.root / "summary.json").read_text())
    assert summary["n_runs"] == 2
    assert set(summary["by_mode"]) == {"existing_skills", "generated_skills"}
    assert len(list((writer.root / "traces").glob("*.json"))) == 2
    manifest = json.loads((writer.root / "run-manifest.json").read_text())
    assert manifest["schema_version"] == "1.1"
    assert len(manifest["catalog_phases"]) == 2


def test_merge_check_only_combines_preflight(tmp_path: Path) -> None:
    base = _phase(tmp_path, "base", "existing_skills", check_only=True)
    generated = _phase(tmp_path, "generated", "generated_skills", check_only=True)
    for phase, mode in ((base, "existing_skills"), (generated, "generated_skills")):
        (phase / "preflight.json").write_text(
            json.dumps([{"case_id": "case-0001-00", "mode": mode, "status": "ready"}]),
            encoding="utf-8",
        )

    results, writer, check_only = merge_phases(
        [base, generated], results_root=tmp_path / "results", eval_id="eval-1",
    )

    assert results == []
    assert check_only is True
    report = json.loads((writer.root / "preflight.json").read_text())
    assert [row["mode"] for row in report] == ["existing_skills", "generated_skills"]


def test_merge_rejects_duplicate_mode(tmp_path: Path) -> None:
    first = _phase(tmp_path, "first", "existing_skills")
    second = _phase(tmp_path, "second", "existing_skills")

    with pytest.raises(ValueError, match="duplicate or empty mode"):
        merge_phases(
            [first, second], results_root=tmp_path / "results", eval_id="eval-1",
        )
