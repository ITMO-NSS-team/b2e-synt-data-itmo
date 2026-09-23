"""Unit tests for the benchmark runner and result files. The stand is never called."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.benchmark.cases import BenchmarkCase
from sim.benchmark.contracts import PROMPT_RENDERER_VERSION, response_contract_hash
from sim.benchmark.execution import ActivatedMode, AgentRequest, AgentTurn
from sim.benchmark.modes import (
    GENERAL_KNOWLEDGE_TOOLS, SKILL_TOOLS, BenchmarkMode, CommonConditions,
    ModeConfig,
)
from sim.benchmark.preflight import PreflightResult
from sim.benchmark.results import ResultWriter, summarize_results
from sim.benchmark.runner import BenchmarkRunner
from sim.fingerprint import RunFingerprint
from tests.fixtures.constants import EMPTY_HASH, EXAMPLE


def ready_case(case_id: str = "case-ready") -> BenchmarkCase:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.update({
        "case_id": case_id, "status": "ready", "employee_id": "123",
        "employee_role": "manager", "snapshot_id": "heimdall-sandbox@test",
        "evaluation_contract": {
            "expected_outcome": "answer",
            "gold_result": [
                {"grade": 10, "employee_count": 3},
                {"grade": 11, "employee_count": 2},
            ],
            "comparison": {
                "ordered": False, "row_key": ["grade"],
                "allow_extra_rows": False, "numeric_absolute_tolerance": 0,
            },
        },
    })
    return BenchmarkCase(Path(f"/authorial/{case_id}.json"), raw)


def mode(name: str = "existing_skills", *, mock: bool = False) -> ModeConfig:
    common = CommonConditions(
        "model", 0.0, "prompt@1", "heimdall-sandbox@test", True, "instant", ()
    )
    enabled = name != BenchmarkMode.GENERAL_KNOWLEDGE
    generated_names = (
        ("generated_headcount",)
        if not mock and name == BenchmarkMode.GENERATED_SKILLS else ()
    )
    return ModeConfig(
        name, SKILL_TOOLS if enabled else GENERAL_KNOWLEDGE_TOOLS,
        None, None, None, None,
        generated_names, common, skills_enabled=enabled, is_mock=mock,
    )


def fingerprint() -> RunFingerprint:
    return RunFingerprint.create(
        agent_config_version="agent@1", prompt_registry_version="prompt@1",
        skill_registry_hash=EMPTY_HASH, model_id="model",
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
            "result": [
                {"grade": 11, "employee_count": 2},
                {"grade": 10, "employee_count": 3},
            ],
            "message": None,
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
            fingerprint=fingerprint().as_dict(),
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
    results = runner.run([ready_case()], {"existing_skills": mode()}, repetitions=2)
    assert len(results) == 2
    assert len({request.metadata["run_id"] for request in executor.requests}) == 2
    for request in executor.requests:
        case = ready_case()
        assert request.query.startswith(case.raw["query"])
        assert "Формат ответа для автоматической проверки" in request.query
        assert '"result"' in request.query
        assert "access_control" not in request.query
        assert "no_data" not in request.query
        assert request.metadata["response_contract_hash"] == response_contract_hash(
            case.raw["response_contract"]
        )
        assert request.metadata["heimdall_access"] == "enabled"
        assert request.metadata["prompt_renderer_version"] == PROMPT_RENDERER_VERSION
        assert request.employee_id == "123"
        assert "gold_result" not in repr(request)
        assert "employee_count\": 3" not in repr(request)
        assert "expected_skills" not in repr(request)
    assert activator.activated == ["existing_skills", "existing_skills"]
    assert activator.deactivated == activator.activated
    assert all(result.score["metrics"]["answer_accuracy"] == 1 for result in results)


def test_runner_marks_tool_free_arm_for_trace_collection() -> None:
    executor, activator = FakeExecutor(), FakeActivator()
    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=activator, executor=executor,
    )

    runner.run(
        [ready_case()],
        {"general_knowledge": mode("general_knowledge")},
    )

    assert executor.requests[0].metadata["heimdall_access"] == "disabled"


def test_runner_treats_blocked_bash_as_completed_when_answer_exists() -> None:
    class DeniedExecutor(FakeExecutor):
        def execute(self, request: AgentRequest) -> AgentTurn:
            turn = super().execute(request)
            return AgentTurn(
                turn.answer, turn.stats, turn.trace, error="['denied:Bash']",
                session_id=turn.session_id, trace_id=turn.trace_id,
                fingerprint=turn.fingerprint,
            )

    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=DeniedExecutor(),
    )
    result = runner.run([ready_case()], {"existing_skills": mode()})[0]
    assert result.status == "completed"
    assert result.response["response"]["error"] is None
    assert result.score["metrics"]["answer_accuracy"] == 1
    assert result.score["reasons"] == []


def test_runner_skips_draft_and_mock_without_activation() -> None:
    executor, activator = FakeExecutor(), FakeActivator()
    states = {
        "existing_skills": "draft_skipped",
        "generated_skills": "mock_skipped",
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
            "existing_skills": mode(),
            "generated_skills": mode("generated_skills", mock=True),
        },
    )
    assert {result.status for result in results} == {"draft_skipped", "mock_skipped"}
    assert not executor.requests
    assert not activator.activated


def test_unstructured_answer_is_pending_not_counted_as_incorrect() -> None:
    class ProseExecutor:
        def execute(self, request: AgentRequest) -> AgentTurn:
            del request
            return AgentTurn(
                "В команде пять сотрудников.",
                trace={"tree": [{"name": "agent.turn", "attributes": {}}]},
                fingerprint=fingerprint().as_dict(),
            )

    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=ProseExecutor(),
    )
    result = runner.run([ready_case()], {"existing_skills": mode()})[0]
    summary = summarize_results([result])
    assert result.status == "normalization_pending"
    assert result.score["metrics"]["answer_accuracy"] is None
    assert summary["n_normalization_pending"] == 1
    assert summary["n_runs"] == 0
    assert summary["review"][0]["run_id"] == result.run_id


def test_runner_marks_fingerprint_drift_without_failing_the_cell_as_wrong() -> None:
    class DriftedExecutor(FakeExecutor):
        def execute(self, request: AgentRequest) -> AgentTurn:
            turn = super().execute(request)
            drifted = dict(turn.fingerprint)
            drifted["model_id"] = "another-model"
            return AgentTurn(
                turn.answer, turn.stats, turn.trace,
                session_id=turn.session_id, trace_id=turn.trace_id,
                fingerprint=drifted,
            )

    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=DriftedExecutor(),
    )
    result = runner.run([ready_case()], {"existing_skills": mode()})[0]
    summary = summarize_results([result])
    assert result.status == "condition_invalid"
    assert result.response["response"]["error"].endswith("model_id")
    assert result.score["metrics"]["answer_accuracy"] is None
    assert summary["n_condition_invalid"] == 1
    assert summary["n_runs"] == 0


def test_runner_keeps_the_full_executor_traceback() -> None:
    class BoomExecutor:
        def execute(self, request: AgentRequest) -> AgentTurn:
            del request
            raise RuntimeError("stand exploded " + ("z" * 80))

    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=BoomExecutor(),
    )
    result = runner.run([ready_case()], {"existing_skills": mode()})[0]
    error = result.response["response"]["error"]
    assert result.status == "unscored"
    assert result.score["metrics"]["answer_accuracy"] is None
    assert "Traceback (most recent call last)" in error
    assert "stand exploded " + ("z" * 80) in error
    assert "RuntimeError" in error
    assert summarize_results([result])["n_unscored"] == 1


def test_missing_trace_is_unscored_even_when_json_matches() -> None:
    class NoTraceExecutor(FakeExecutor):
        def execute(self, request: AgentRequest) -> AgentTurn:
            turn = super().execute(request)
            return AgentTurn(
                turn.answer, turn.stats, None,
                session_id=turn.session_id, trace_id=turn.trace_id,
                fingerprint=turn.fingerprint,
            )

    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=NoTraceExecutor(),
    )
    result = runner.run([ready_case()], {"existing_skills": mode()})[0]
    assert result.status == "unscored"
    assert "trace unavailable" in result.response["response"]["error"]
    assert result.score["metrics"]["answer_accuracy"] is None
    assert summarize_results([result])["n_runs"] == 0


def test_result_writer_keeps_trace_separate_and_writes_summary(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path, "eval-1")
    runner = BenchmarkRunner(
        "eval-1",
        preflight=lambda case, selected: PreflightResult(
            case.case_id, selected.name, "ready", fingerprint()
        ),
        activator=FakeActivator(), executor=FakeExecutor(), writer=writer,
    )
    results = runner.run([ready_case()], {"existing_skills": mode()})
    response = json.loads(writer.responses_path.read_text(encoding="utf-8"))
    score = json.loads(writer.scores_path.read_text(encoding="utf-8"))
    summary = json.loads((writer.root / "summary.json").read_text(encoding="utf-8"))
    writer.write_manifest({"schema_version": "1.0", "eval_id": "eval-1"})
    assert response["response"]["raw_answer"]
    assert response["case_snapshot"]["evaluation_contract"]
    assert response["case_snapshot"]["original_query"]
    assert response["case_snapshot"]["rendered_query"]
    assert response["case_snapshot"]["response_contract_hash"].startswith("sha256:")
    assert (writer.root / response["trace_path"]).is_file()
    assert score["metrics"]["answer_accuracy"] == 1
    assert summary["by_mode"]["existing_skills"]["answer_accuracy"] == 1
    assert (writer.root / "run-manifest.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        ResultWriter(tmp_path, "eval-1")
    assert summarize_results(results)["n_runs"] == 1
