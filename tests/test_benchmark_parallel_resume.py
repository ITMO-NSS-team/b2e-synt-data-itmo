"""Bounded read-only parallelism and durable per-cell resume."""
import json
import threading
import pytest
from sim.benchmark.results import ResultWriter
from sim.benchmark.runner import BenchmarkRunner
from tests.test_benchmark_runner import FakeActivator, FakeExecutor, ready_case, mode
from sim.benchmark.preflight import PreflightResult


def _runner(writer, executor, activator=None):
    return BenchmarkRunner("test", preflight=lambda case, mode: PreflightResult(case.case_id, mode.name, "ready"),
                           activator=activator or FakeActivator(), executor=executor, writer=writer)


def test_parallel_rejects_an_activator_that_might_mutate_global_state(tmp_path):
    runner = _runner(ResultWriter(tmp_path, "test"), FakeExecutor())
    with pytest.raises(ValueError, match="read-only"):
        runner.run([ready_case()], [mode()], workers=2)


def test_parallel_executes_overlapping_calls_but_writes_each_result_once(tmp_path):
    barrier = threading.Barrier(2)
    class Executor(FakeExecutor):
        def execute(self, request):
            barrier.wait(timeout=3)
            return super().execute(request)
    executor = Executor()
    activator = FakeActivator()
    activator.supports_concurrent = True
    writer = ResultWriter(tmp_path, "test")
    results = _runner(writer, executor, activator).run(
        [ready_case(f"case-{i:04d}-00") for i in range(4)], [mode()], workers=2,
    )
    assert len(results) == len(executor.requests) == 4
    raw = [json.loads(line) for line in writer.responses_path.read_text().splitlines()]
    assert len({row["run_id"] for row in raw}) == 4
    assert len(writer.scores_path.read_text().splitlines()) == 4


def test_resume_does_not_repeat_saved_cells(tmp_path):
    writer = ResultWriter(tmp_path, "test")
    cases = [ready_case(f"case-{i:04d}-00") for i in range(4)]
    first = _runner(writer, FakeExecutor())
    for case in cases[:2]:
        writer.write(first.run_one(case, mode(), 1))
    executor = FakeExecutor()
    resumed = ResultWriter(tmp_path, "test", resume=True)
    results = _runner(resumed, executor).run(cases, [mode()])
    assert len(results) == 4
    assert len(executor.requests) == 2
    assert len(resumed.responses_path.read_text().splitlines()) == 4
    assert len(resumed.scores_path.read_text().splitlines()) == 4


def test_resume_rejects_incompatible_conditions(tmp_path):
    writer = ResultWriter(tmp_path, "test")
    manifest = {"eval_id": "test", "case_ids": [], "modes": [{
        "name": "skills_disabled",
        "common": {"model_id": "one", "hr_employee_ids": ("hr-1",)},
        "tool_subset": ("mcp_query",),
        "generated_skill_names": ("skill-v1-0-0",),
    }]}
    writer.write_manifest(manifest)
    resumed = ResultWriter(tmp_path, "test", resume=True)
    resumed.write_manifest(manifest)
    saved = json.loads((writer.root / "run-manifest.json").read_text())
    assert saved["modes"][0]["tool_subset"] == ["mcp_query"]
    assert saved["modes"][0]["generated_skill_names"] == ["skill-v1-0-0"]
    assert saved["modes"][0]["common"]["hr_employee_ids"] == ["hr-1"]
    manifest["modes"][0]["common"]["model_id"] = "two"
    with pytest.raises(ValueError, match="resume manifest differs"):
        resumed.write_manifest(manifest)


def test_infrastructure_breaker_preserves_checkpoint_and_stops_submitting(tmp_path):
    from sim.benchmark.execution import AgentTurn
    class Failed(FakeExecutor):
        def execute(self, request):
            self.requests.append(request)
            return AgentTurn(answer="", error="provider unavailable")
    executor = Failed()
    writer = ResultWriter(tmp_path, "test")
    cases = [ready_case(f"case-{i:04d}-00") for i in range(8)]
    with pytest.raises(RuntimeError, match="consecutive infrastructure failures"):
        _runner(writer, executor).run(cases, [mode()], max_consecutive_unscored=2)
    assert len(executor.requests) == 2
    assert len(writer.responses_path.read_text().splitlines()) == 2
    assert (writer.root / "summary.json").is_file()
