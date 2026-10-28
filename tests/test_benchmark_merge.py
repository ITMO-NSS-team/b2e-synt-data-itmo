from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.benchmark.merge import (
    _selection_summary, _validate_generated_plan_execution, merge_phases,
)
from sim.benchmark.results import ResultWriter


def _phase(
    root: Path, name: str, mode: str, *, check_only: bool = False,
    case_id: str = "case-0001-00",
) -> Path:
    phase = root / name
    phase.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0",
        "eval_id": "eval-1",
        "check_only": check_only,
        "cases_path": "/cases",
        "case_ids": [case_id],
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


def test_resume_rebuilds_partial_final_without_duplicate_rows(tmp_path: Path) -> None:
    phase = _phase(tmp_path, "base", "existing_skills")
    _run(phase, "existing_skills")
    results, writer, _ = merge_phases(
        [phase], results_root=tmp_path / "results", eval_id="eval-1",
    )
    response = writer.responses_path.read_text()
    writer.responses_path.write_text(response + response)
    (writer.root / "summary.json").unlink()

    resumed, final, _ = merge_phases(
        [phase], results_root=tmp_path / "results", eval_id="eval-1", resume=True,
    )

    assert len(results) == len(resumed) == 1
    assert final.responses_path.read_text() == response
    assert (final.root / "summary.json").is_file()
    assert (final.root / "report.html").is_file()
    assert not list((tmp_path / "results").glob(".merge-*"))


def test_resume_keeps_unchanged_final_files(tmp_path: Path) -> None:
    phase = _phase(tmp_path, "base", "existing_skills")
    _run(phase, "existing_skills")
    _results, writer, _ = merge_phases(
        [phase], results_root=tmp_path / "results", eval_id="eval-1",
    )
    original = (writer.root / "summary.json").stat().st_mtime_ns
    (writer.root / ".telemetry-published").write_text("published\n")

    merge_phases([phase], results_root=tmp_path / "results", eval_id="eval-1", resume=True)

    assert (writer.root / "summary.json").stat().st_mtime_ns == original
    assert (writer.root / ".telemetry-published").read_text() == "published\n"
    assert len(writer.responses_path.read_text().splitlines()) == 1


def test_resume_refreshes_final_after_a_phase_gains_a_cell(tmp_path: Path) -> None:
    phase = _phase(tmp_path, "base", "existing_skills")
    manifest_path = phase / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["case_ids"].append("case-0002-00")
    manifest_path.write_text(json.dumps(manifest))
    _run(phase, "existing_skills")
    merge_phases([phase], results_root=tmp_path / "results", eval_id="eval-1")
    raw = json.loads((phase / "responses.jsonl").read_text())
    raw["case_id"] = "case-0002-00"
    raw["run_id"] = "eval-1-case-0002-00-existing_skills-01"
    raw["comparison_group_id"] = "eval-1-case-0002-00-01"
    raw.pop("trace_path")
    with (phase / "responses.jsonl").open("a") as handle:
        handle.write(json.dumps(raw) + "\n")
    with (phase / "scores.jsonl").open("a") as handle:
        handle.write(json.dumps({"run_id": raw["run_id"], "metrics": {"answer_accuracy": 1}}) + "\n")
    results, writer, _ = merge_phases(
        [phase], results_root=tmp_path / "results", eval_id="eval-1", resume=True,
    )

    assert len(results) == 2
    assert len(writer.responses_path.read_text().splitlines()) == 2
    assert json.loads((writer.root / "summary.json").read_text())["n_runs"] == 2


def test_failed_rebuild_keeps_previous_final(tmp_path: Path, monkeypatch) -> None:
    phase = _phase(tmp_path, "base", "existing_skills")
    _run(phase, "existing_skills")
    _results, writer, _ = merge_phases(
        [phase], results_root=tmp_path / "results", eval_id="eval-1",
    )
    original = writer.responses_path.read_bytes()
    public = tmp_path / "public"
    monkeypatch.setenv("BENCHMARK_REPORT_DIR", str(public))

    def fail_summary(*_args, **_kwargs):
        raise RuntimeError("interrupted merge")

    monkeypatch.setattr(ResultWriter, "write_summary", fail_summary)
    with pytest.raises(RuntimeError, match="interrupted merge"):
        merge_phases([phase], results_root=tmp_path / "results", eval_id="eval-1", resume=True)

    assert writer.responses_path.read_bytes() == original
    assert (writer.root / "summary.json").is_file()
    assert not public.exists()
    assert not list((tmp_path / "results").glob(".merge-*"))


def test_failed_promotion_and_rollback_preserve_backup(tmp_path: Path, monkeypatch) -> None:
    phase = _phase(tmp_path, "base", "existing_skills")
    _run(phase, "existing_skills")
    _results, writer, _ = merge_phases(
        [phase], results_root=tmp_path / "results", eval_id="eval-1",
    )
    (writer.root / "summary.json").unlink()
    original_rename = Path.rename

    def fail_rename(path, target):
        if path.name == "eval-1" and path.parent.name.startswith(".merge-"):
            raise OSError("promotion failed")
        if path.name.startswith(".backup-"):
            raise OSError("rollback failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="rollback failed"):
        merge_phases([phase], results_root=tmp_path / "results", eval_id="eval-1", resume=True)

    backups = list((tmp_path / "results").glob(".backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "run-manifest.json").is_file()


def test_generated_plan_checks_only_modes_assigned_to_each_group() -> None:
    plan = {
        "generated_modes": ["existing_plus_generated", "existing_plus_catalog"],
        "groups": [
            {"variant_id": "one", "modes": ["existing_plus_generated"]},
            {"variant_id": "catalog--baseline", "modes": ["existing_plus_catalog"]},
        ],
    }
    manifests = [
        {"modes": [{"name": "existing_plus_generated@one"}]},
        {"modes": [{"name": "existing_plus_catalog@catalog--baseline"}]},
    ]
    _validate_generated_plan_execution(manifests, plan)


def test_catalog_coverage_is_reported_separately_from_case_selection() -> None:
    plan = {
        "source_case_ids": ["a", "b"], "eligible_case_ids": ["a", "b"],
        "selected_case_ids": ["a", "b"],
        "groups": [{"scope": "catalog", "source": "baseline",
                    "case_ids": ["a", "b"], "matched_case_ids": ["a"]}],
    }

    selection = _selection_summary(plan)["selection"]

    assert selection["coverage_rate"] == 1.0
    assert selection["catalog_coverage"] == {
        "baseline": {"n_cases": 2, "n_target_cases": 1},
    }


@pytest.mark.parametrize("check_only", [False, True])
def test_zero_coverage_writes_report_without_run_cells(tmp_path: Path, check_only: bool) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "cases_source": "/cases",
        "source_case_ids": ["case-0001-00"],
        "eligible_case_ids": [], "selected_case_ids": [], "groups": [],
        "excluded": [{"case_id": "case-0001-00", "status": "generated_skill_unavailable_excluded"}],
        "unmatched_artifacts": [{"skill_name": "alpha", "source": "baseline", "version": "1.0.0"}],
    }), encoding="utf-8")

    results, writer, is_check = merge_phases(
        [], results_root=tmp_path / "results", eval_id="eval-1",
        generated_plan=plan, check_only=check_only,
    )

    assert results == []
    assert is_check == check_only
    if check_only:
        assert json.loads((writer.root / "preflight.json").read_text()) == []
    else:
        assert json.loads((writer.root / "summary.json").read_text())["selection"]["n_unmatched_artifacts"] == 1
        assert "alpha" in (writer.root / "report.html").read_text()
    assert merge_phases(
        [], results_root=tmp_path / "results", eval_id="eval-1",
        generated_plan=plan, check_only=check_only, resume=True,
    )[0] == []


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


def test_merge_check_only_does_not_add_excluded_cases_to_preflight(tmp_path: Path) -> None:
    phase = _phase(
        tmp_path, "generated", "generated_skills", check_only=True,
    )
    ready = {
        "case_id": "case-0001-00",
        "mode": "generated_skills",
        "status": "ready",
    }
    (phase / "preflight.json").write_text(
        json.dumps([ready]), encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "cases_source": "/cases",
        "source_case_ids": ["case-0001-00", "case-0002-00"],
        "eligible_case_ids": ["case-0001-00"],
        "selected_case_ids": ["case-0001-00"],
        "groups": [],
        "excluded": [{
            "case_id": "case-0002-00",
            "status": "generated_skill_unavailable_excluded",
            "reason": "generated skill is unavailable: missing",
            "expected_skills": ["missing"],
        }],
    }), encoding="utf-8")

    _results, writer, _check = merge_phases(
        [phase], results_root=tmp_path / "results", eval_id="eval-1",
        generated_plan=plan,
    )

    report = json.loads((writer.root / "preflight.json").read_text())
    assert len(report) == 1


def test_merge_rejects_duplicate_mode(tmp_path: Path) -> None:
    first = _phase(tmp_path, "first", "existing_skills")
    second = _phase(tmp_path, "second", "existing_skills")

    with pytest.raises(ValueError, match="duplicate case/mode"):
        merge_phases(
            [first, second], results_root=tmp_path / "results", eval_id="eval-1",
        )


def test_merge_accepts_same_generated_mode_for_disjoint_case_groups(tmp_path: Path) -> None:
    first = _phase(tmp_path, "first", "generated_skills", case_id="case-0001-00")
    second = _phase(tmp_path, "second", "generated_skills", case_id="case-0002-00")
    _run(first, "generated_skills", with_trace=False)
    # The helper writes case-0001 identifiers; make this phase represent its
    # declared disjoint case before merging.
    run_id = "eval-1-case-0002-00-generated_skills-01"
    raw = {
        "run_id": run_id,
        "comparison_group_id": "eval-1-case-0002-00-01",
        "case_id": "case-0002-00",
        "mode": "generated_skills",
        "repetition": 1,
        "status": "completed",
        "response": {"raw_answer": "{}", "error": None},
    }
    (second / "responses.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    (second / "scores.jsonl").write_text(
        json.dumps({"run_id": run_id, "metrics": {"answer_accuracy": 1}, "reasons": []}) + "\n",
        encoding="utf-8",
    )

    results, _writer, _check = merge_phases(
        [first, second], results_root=tmp_path / "results", eval_id="eval-1",
    )

    assert {row.case_id for row in results} == {"case-0001-00", "case-0002-00"}


def test_merge_reports_generated_selection_without_creating_fake_runs(tmp_path: Path) -> None:
    base = _phase(tmp_path, "base", "existing_skills")
    _run(base, "existing_skills", with_trace=False)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "cases_source": "/cases",
        "source_case_ids": ["case-0001-00", "case-0002-00"],
        "eligible_case_ids": ["case-0001-00"],
        "selected_case_ids": ["case-0001-00"],
        "groups": [],
        "excluded": [{
            "case_id": "case-0002-00",
            "status": "generated_skill_unavailable_excluded",
            "reason": "generated skill is unavailable: missing",
            "expected_skills": ["missing"],
        }],
    }), encoding="utf-8")

    results, writer, _check = merge_phases(
        [base], results_root=tmp_path / "results", eval_id="eval-1",
        generated_plan=plan,
    )

    assert len(results) == 1
    summary = json.loads((writer.root / "summary.json").read_text())
    assert summary["n_skipped"] == 0
    assert summary["selection"] == {
        "n_source_cases": 2,
        "n_eligible_cases": 1,
        "n_selected_cases": 1,
        "n_excluded_cases": 1,
        "n_partially_covered_cases": 0,
            "n_unmatched_artifacts": 0,
        "coverage_rate": 0.5,
        "excluded_by_reason": {"generated_skill_unavailable_excluded": 1},
    }


def test_merge_accepts_two_generated_variants_for_the_same_case(tmp_path: Path) -> None:
    baseline = _phase(tmp_path, "baseline", "existing_skills")
    first = _phase(tmp_path, "first", "generated_skills@alpha_1")
    second = _phase(tmp_path, "second", "generated_skills@alpha_2")
    _run(baseline, "existing_skills", with_trace=False)
    _run(first, "generated_skills@alpha_1", with_trace=False)
    _run(second, "generated_skills@alpha_2", with_trace=False)

    results, writer, _check = merge_phases(
        [baseline, first, second], results_root=tmp_path / "results", eval_id="eval-1",
    )

    assert {row.mode for row in results} == {
        "existing_skills", "generated_skills@alpha_1", "generated_skills@alpha_2",
    }
    summary = json.loads((writer.root / "summary.json").read_text())
    assert set(summary["by_mode"]) == {
        "existing_skills", "generated_skills@alpha_1", "generated_skills@alpha_2",
    }
    assert summary["comparisons"][
        "generated_skills@alpha_1_vs_existing_skills"
    ]["n_pairs"] == 1
    assert summary["comparisons"][
        "generated_skills@alpha_2_vs_existing_skills"
    ]["n_pairs"] == 1


def test_merge_rejects_missing_planned_generated_variant(tmp_path: Path) -> None:
    generated = _phase(tmp_path, "generated", "generated_skills@alpha_1")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "groups": [
            {"variant_id": "alpha_1"},
            {"variant_id": "alpha_2"},
        ],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="missing variants: generated_skills@alpha_2"):
        merge_phases(
            [generated], results_root=tmp_path / "results", eval_id="eval-1",
            generated_plan=plan,
        )
