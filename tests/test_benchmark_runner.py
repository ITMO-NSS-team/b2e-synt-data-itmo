"""Unit tests for the benchmark runner and result files. The stand is never called."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.benchmark.cases import BenchmarkCase
from sim.benchmark.execution import ActivatedMode, AgentRequest, AgentTurn
from sim.benchmark.modes import DATA_TOOLS, SKILL_TOOLS, BenchmarkMode, CommonConditions, ModeConfig
from sim.benchmark.preflight import PreflightResult
from sim.benchmark.results import ResultWriter, summarize_results
from sim.benchmark.runner import BenchmarkRunner
from sim.fingerprint import RunFingerprint

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "benchmarking/cases/case-0001.json"


def ready_case(case_id: str = "case-ready") -> BenchmarkCase:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update({
        "case_id": case_id, "status": "ready", "employee_id": "123",
        "employee_role": "manager", "snapshot_id": "heimdall-sandbox@test",
        "gold_answer": {
            "outcome": "answer",
            "rows": [
                {"grade": 10, "employee_count": 3},
                {"grade": 11, "employee_count": 2},
            ],
        },
    })
    return BenchmarkCase(Path(f"/authorial/{case_id}.json"), raw)


def mode(name: str = "heimdall_skills", *, mock: bool = False) -> ModeConfig:
    common = CommonConditions(
        "model", 0.0, "prompt@1", "heimdall-sandbox@test", True, "instant", ()
    )
    enabled = name != BenchmarkMode.SKILLS_DISABLED
    generated_names = (
        ("generated_headcount",)
        if not mock and name == BenchmarkMode.GENERATED_SKILL else ()
    )
    return ModeConfig(
        name, SKILL_TOOLS if enabled else DATA_TOOLS, None, None, None, None,
        generated_names, common, skills_enabled=enabled, is_mock=mock,
    )


def fingerprint() -> RunFingerprint:
    return RunFingerprint.create(
        agent_config_version="agent@1", prompt_registry_version="prompt@1",
        skill_registry_hash="sha256:" + "0" * 64, model_id="model",
        temperature=0, data_snapshot_hash="heimdall-sandbox@test",
        traps_enabled=True, latency_profile="instant", hr_employee_ids=(),
    )


class FakeActivator:
    def __init__(self) -> None:
        self.activated: list[str] = []
        self.deactivated: list[str] = []

    def activate(self, value: ModeConfig) -> ActivatedMode:
        self.activated.append(value.name)
        return ActivatedMode(f"{value.name}@1", value.catalog_hash)

    def deactivate(self, value: ModeConfig) -> None:
        self.deactivated.append(value.name)


class FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    def execute(self, request: AgentRequest) -> AgentTurn:
        self.requests.append(request)
        answer = json.dumps({
            "outcome": "answer",
            "rows": [
                {"grade": 11, "employee_count": 2},
                {"grade": 10, "employee_count": 3},
            ],
        })
        trace = {"tree": [{
            "name": "agent.turn",
            "attributes": {
                "b2e.turn.duration_ms": 1200,
                "b2e.turn.tool_time_ms": 300,
            },
            "children": [{
                "name": "heimdall.mcp_query",
                "attributes": {
                    "b2e.heimdall.endpoint": "mcp_query",
                    "b2e.http.status": 200,
                },
            }],
        }]}
        return AgentTurn(
            answer,
            {"tool_calls": 2, "heimdall_calls": 1, "total_tokens": 321,
             "latency_ms": 1500},
            trace, session_id=f"session-{len(self.requests)}", trace_id="trace-1",
        )


def test_runner_never_sends_gold_or_expected_skill_to_agent() -> None:
    executor, activator = FakeExecutor(), FakeActivator()
    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=activator, executor=executor,
    )
    results = runner.run([ready_case()], {"heimdall_skills": mode()}, repetitions=2)
    assert len(results) == 2
    assert len({request.metadata["run_id"] for request in executor.requests}) == 2
    for request in executor.requests:
        assert request.query == ready_case().raw["query"]
        assert request.employee_id == "123"
        assert "gold_answer" not in repr(request)
        assert "expected_skills" not in repr(request)
    assert activator.activated == ["heimdall_skills", "heimdall_skills"]
    assert activator.deactivated == activator.activated
    assert all(result.score["metrics"]["answer_accuracy"] == 1 for result in results)


def test_runner_skips_draft_and_mock_without_activation() -> None:
    executor, activator = FakeExecutor(), FakeActivator()
    states = {
        "heimdall_skills": "draft_skipped",
        "generated_skill": "mock_skipped",
    }
    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, states[selected.name]
        ),
        activator=activator, executor=executor,
    )
    results = runner.run(
        [ready_case()],
        {
            "heimdall_skills": mode(),
            "generated_skill": mode("generated_skill", mock=True),
        },
    )
    assert {result.status for result in results} == {"draft_skipped", "mock_skipped"}
    assert not executor.requests
    assert not activator.activated


def test_unstructured_answer_is_pending_not_counted_as_incorrect() -> None:
    class ProseExecutor:
        def execute(self, request: AgentRequest) -> AgentTurn:
            del request
            return AgentTurn("В команде пять сотрудников.")

    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=ProseExecutor(),
    )
    result = runner.run([ready_case()], {"heimdall_skills": mode()})[0]
    assert result.status == "normalization_pending"
    assert result.score["metrics"]["answer_accuracy"] is None
    assert summarize_results([result])["n_normalization_pending"] == 1


def test_result_writer_keeps_trace_separate_and_writes_summary(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path, "eval-1")
    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=FakeExecutor(), writer=writer,
    )
    results = runner.run([ready_case()], {"heimdall_skills": mode()})
    response = json.loads(writer.responses_path.read_text(encoding="utf-8"))
    score = json.loads(writer.scores_path.read_text(encoding="utf-8"))
    summary = json.loads((writer.root / "summary.json").read_text(encoding="utf-8"))
    assert response["response"]["raw_answer"]
    assert response["case_snapshot"]["gold_answer"]
    assert (writer.root / response["trace_path"]).is_file()
    assert score["metrics"]["answer_accuracy"] == 1
    assert summary["by_mode"]["heimdall_skills"]["answer_accuracy"] == 1
    with pytest.raises(ValueError, match="already exists"):
        ResultWriter(tmp_path, "eval-1")
    assert summarize_results(results)["n_runs"] == 1
