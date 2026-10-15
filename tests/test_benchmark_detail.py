"""Matched detail catalogs and paired estimates; no Docker or model calls."""
import json
from pathlib import Path
import pytest
from heimdall.skills.registry import Registry
from sim.benchmark.detail_plan import build_detail_plan, LEVELS
from sim.benchmark.detail_report import analyze_detail
from tests.test_benchmark_generated_plan import _case, _skill
from tests.fixtures.constants import SCHEMA


@pytest.fixture
def inputs(tmp_path):
    cases, skills = tmp_path / "cases", tmp_path / "skills"
    cases.mkdir()
    _case(cases, "case-0101-00", ["alpha"])
    _case(cases, "case-0101-01", ["alpha"])
    _case(cases, "case-0102-00", ["beta"])
    for mi, method in enumerate(("no-context", "compact")):
        for li, level in enumerate(LEVELS):
            for canonical in ("alpha", "beta"):
                _skill(skills / method / level, f"{canonical}-{method}-{li+1:010x}-{mi+1:010x}")
    return cases, skills


def test_detail_catalog_contains_all_skills_but_only_one_method_and_level(inputs, tmp_path):
    cases, skills = inputs
    original = {p: p.read_bytes() for p in cases.glob("*.json")}
    plan = build_detail_plan(cases, skills, tmp_path / "plan", schema_path=SCHEMA)
    assert len(plan["groups"]) == 6
    assert len(plan["selected_case_ids"]) == 3
    for group in plan["groups"]:
        assert group["case_ids"] == plan["selected_case_ids"]
        assert len(Registry.load(group["catalog_path"]).active()) == 2
        assert group["expected_skills"] == ["alpha", "beta"]
        for raw in map(json.loads, Path(group["cases_path"]).read_text().splitlines()):
            canonical = plan["case_skills"][raw["case_id"]][0]
            assert raw["expected_skills"] == [group["skill_map"][canonical]]
    assert all(p.read_bytes() == content for p, content in original.items())


def test_missing_level_excludes_case_from_every_arm(inputs, tmp_path):
    cases, skills = inputs
    for path in (skills / "compact" / "medium").glob("beta*/SKILL.md"):
        path.unlink()
    # Its resources are .txt and are not catalog skill files.
    plan = build_detail_plan(cases, skills, tmp_path / "plan", schema_path=SCHEMA)
    assert plan["selected_case_ids"] == ["case-0101-00", "case-0101-01"]
    assert plan["excluded"][0]["case_id"] == "case-0102-00"
    assert plan["excluded"][0]["missing_by_arm"] == {"compact/medium": ["beta"]}
    assert all(group["case_ids"] == plan["selected_case_ids"] for group in plan["groups"])


def test_duplicate_generations_are_not_selected_silently(inputs, tmp_path):
    cases, skills = inputs
    _skill(skills / "compact" / "medium", "alpha-compact-aaaaaaaaaa-bbbbbbbbbb")
    with pytest.raises(ValueError, match="multiple generations"):
        build_detail_plan(cases, skills, tmp_path / "plan", schema_path=SCHEMA)


def test_resume_rejects_changed_artifact_or_cases(inputs, tmp_path):
    cases, skills = inputs
    root = tmp_path / "plan"
    first = build_detail_plan(cases, skills, root, schema_path=SCHEMA)
    assert build_detail_plan(cases, skills, root, schema_path=SCHEMA, resume=True) == first
    path = next(skills.rglob("SKILL.md"))
    path.write_text(path.read_text() + "changed source")
    with pytest.raises(ValueError, match="resume inputs differ"):
        build_detail_plan(cases, skills, root, schema_path=SCHEMA, resume=True)


def test_paired_report_keeps_methods_separate_and_uses_skill_clusters():
    case_skills = {"a1": ["alpha"], "a2": ["alpha"], "a3": ["alpha"], "b1": ["beta"]}
    groups = [{"variant_id": method + "--" + level, "generation_method": method, "detail_level": level}
              for method in ("compact", "no-context") for level in ("abstract", "detailed")]
    plan = {"groups": groups, "methods": ["compact", "no-context"], "detail_levels": ["abstract", "detailed"],
            "selected_case_ids": list(case_skills), "source_case_ids": list(case_skills), "excluded": [],
            "case_skills": case_skills, "common_skills": ["alpha", "beta"]}
    responses, scores = [], []
    for group in groups:
        for case in case_skills:
            success = int((group["detail_level"] == "detailed") == (case.startswith("a")))
            if group["generation_method"] == "no-context":
                success = 0
            run_id = group["variant_id"] + "-" + case
            responses.append({"run_id": run_id, "mode": "generated_skills@" + group["variant_id"], "case_id": case, "repetition": 1, "status": "completed"})
            scores.append({"run_id": run_id, "metrics": {"answer_accuracy": success}})
    report = analyze_detail(plan, responses, scores)
    compact = report["paired_comparisons"]["compact"]["detailed_vs_abstract"]
    assert compact["n_pairs"] == 4 and compact["n_skill_clusters"] == 2
    assert compact["accuracy_delta_pp"] == 50
    assert compact["macro_skill_accuracy_delta_pp"] == 0
    assert compact["macro_skill_delta_ci95_pp"] == [-100, 100]
    assert report["paired_comparisons"]["no-context"]["detailed_vs_abstract"]["accuracy_delta_pp"] == 0
    responses[0]["status"] = "unscored"
    scores[0]["metrics"]["answer_accuracy"] = None
    report = analyze_detail(plan, responses, scores)
    assert report["paired_comparisons"]["compact"]["detailed_vs_abstract"]["n_pairs"] == 3
    assert report["by_detail"]["compact"]["abstract"]["score_coverage"] == .75


def test_balanced_sample_keeps_all_skills_and_same_cases_in_each_arm(inputs, tmp_path):
    cases, skills = inputs
    plan = build_detail_plan(cases, skills, tmp_path / "balanced", schema_path=SCHEMA, cases_per_skill=1)
    assert len(plan["selected_case_ids"]) == 2
    assert {name for case in plan["case_skills"].values() for name in case} == {"alpha", "beta"}
    assert plan["case_selection"]["strategy"] == "balanced_per_skill"
    assert all(group["case_ids"] == plan["selected_case_ids"] for group in plan["groups"])


def test_report_files_preserve_results_and_trace_index(tmp_path):
    from sim.benchmark.detail_report import write_detail_report
    groups = [{"variant_id": "compact--" + level, "generation_method": "compact", "detail_level": level}
              for level in ("abstract", "detailed")]
    plan = {"experiment": "prompt_detail", "groups": groups, "methods": ["compact"],
            "detail_levels": ["abstract", "detailed"], "case_skills": {"one": ["alpha"]},
            "source_case_ids": ["one"], "selected_case_ids": ["one"], "excluded": [], "common_skills": ["alpha"]}
    conditions = {"model_id": "model", "harness": "test"}
    modes = [{"name": "generated_skills@" + group["variant_id"], "common": conditions} for group in groups]
    manifest = {"generated_plan": plan, "eval_id": "test", "modes": modes, "repetitions": 1, "workers": 2,
                "live_stand": {"skill_registry_hash": "hash"}}
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text('{"comparisons": {}}')
    responses = [{"run_id": str(i), "case_id": "one", "mode": mode["name"], "repetition": 1,
                  "status": "completed", "trace_id": "abc", "trace_path": "traces/abc.json"}
                 for i, mode in enumerate(modes)]
    scores = [{"run_id": str(i), "metrics": {"answer_accuracy": i, "total_tokens": 123}} for i in range(2)]
    (tmp_path / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in responses))
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores))
    original = (tmp_path / "responses.jsonl").read_bytes()
    report = write_detail_report(tmp_path, phoenix_url="https://localhost/phoenix", project_id="UHJvamVjdDoy")
    assert report["complete_generated_matrix"]
    assert report["paired_comparisons"]["compact"]["detailed_vs_abstract"]["accuracy_delta_pp"] == 100
    assert (tmp_path / "responses.jsonl").read_bytes() == original
    assert all((tmp_path / name).is_file() for name in ("detail-report.html", "detail-report.md", "detail-report.json", "detail-runs.csv"))
    assert "/projects/UHJvamVjdDoy/traces/abc" in (tmp_path / "detail-runs.csv").read_text(encoding="utf-8-sig")


def test_cancelled_progress_stops_watch_without_claiming_final_report(tmp_path):
    from sim.benchmark.detail_progress import write_progress
    plan = tmp_path / "plans" / "cancelled-study" / "plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"selected_case_ids": ["one"], "groups": []}))
    results = tmp_path / "results"
    phase = results / ".phases" / "cancelled-study"
    phase.mkdir(parents=True)
    (phase / "experiment-status.json").write_text(json.dumps({
        "status": "cancelled", "reason": "operator <stop>",
    }))
    stopped, page, count = write_progress(plan, results)
    document = page.read_text(encoding="utf-8")
    assert stopped and count == 0
    assert 'http-equiv="refresh"' not in document
    assert "operator &lt;stop&gt;" in document
    assert 'href="cancelled-study/detail-report.html"' not in document
    assert not (results / "cancelled-study").exists()
