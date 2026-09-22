"""Remote runner is read-only until session creation; no live network in tests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.benchmark import cli, remote
from sim.benchmark.execution import AgentTurn
from sim.benchmark.modes import GENERAL_KNOWLEDGE_TOOLS, SKILL_TOOLS
from tests.fixtures.constants import EXAMPLE


def reports(case):
    raw = case.raw
    on = {
        "model_id": "server-model", "temperature": 0.0,
        "tool_subset": list(SKILL_TOOLS), "code_execution": "forbidden",
        "conversation_mode": "stateless",
    }
    general = dict(on)
    general["tool_subset"] = list(GENERAL_KNOWLEDGE_TOOLS)
    live = {
        "schema_version": "1.0",
        "refs": {
            "general_knowledge": "benchmark_general@1",
            "existing_skills": "agent_config@2",
        },
        "configs": {
            "benchmark_general@1": general,
            "agent_config@2": on,
        },
        "prompt_versions": {
            "benchmark_general@1": "system_prompt@1",
            "agent_config@2": "system_prompt@1",
        },
        "skill_registry_hash": raw["skill_registry_hash"],
        "emulator": {"data_snapshot_hash": raw["snapshot_id"], "traps_enabled": True,
                     "latency_profile": "instant", "hr_employee_ids": []},
    }
    return {
        "live": live,
        "scopes": {raw["employee_id"]: {"employee_id": raw["employee_id"], "role": raw["employee_role"]}},
    }, {
        "emulator": deepcopy(live["emulator"]), "snapshot_id": raw["snapshot_id"],
        "validated": True, "skill_count": 10, "catalog_hash": "sha256:" + "a" * 64,
        "catalog_path": "/app/heimdall-skills", "model_catalog_hash": "sha256:" + "b" * 64,
    }


def test_defaults_use_one_case_current_config_and_phoenix():
    args = remote.parser().parse_args(["--cases", str(EXAMPLE)])
    assert (args.limit, args.repetitions, args.modes) == (
        1, 1, "general_knowledge,existing_skills",
    )
    assert args.config_ref == "agent_config"
    assert args.trace_backend == "phoenix"
    assert args.trace_timeout == 300.0
    assert args.env_file == "deploy/.env"
    assert args.emulator_control_url == "http://127.0.0.1:8081"
    assert args.phoenix_control_url == "http://127.0.0.1:6006"
    assert args.phoenix_project == "b2e-itmo"
    assert not hasattr(args, "trace_attempts")


def test_readonly_report_reuses_pinned_config_without_local_data():
    cases = cli.load_cases_path(EXAMPLE)
    report, catalog = reports(cases[0])
    prepared = remote.prepare_remote(None, cases, report, catalog)
    assert set(prepared.selected_modes) == {"general_knowledge", "existing_skills"}
    mode = prepared.selected_modes["existing_skills"]
    assert mode.common.model_id == "server-model"
    assert mode.catalog_path is None
    assert mode.catalog_hash == catalog["catalog_hash"]
    assert prepared.activator.activate(mode).config_ref == "agent_config@2"


@pytest.mark.parametrize("change, message", [
    ("snapshot", "snapshot_id differs"),
    ("registry", "skill_registry_hash differs"),
    ("role", "role/identity mismatch"),
    ("catalog", "was not validated"),
    ("drift", "changed during discovery"),
    ("tools", "tool_subset differs"),
])
def test_preflight_rejects_mismatches(change, message):
    cases = cli.load_cases_path(EXAMPLE)
    report, catalog = reports(cases[0])
    if change == "snapshot":
        cases[0].raw["snapshot_id"] = "wrong"
    elif change == "registry":
        cases[0].raw["skill_registry_hash"] = "wrong"
    elif change == "role":
        report["scopes"][cases[0].raw["employee_id"]]["role"] = "hr"
    elif change == "catalog":
        catalog["validated"] = False
    elif change == "drift":
        catalog["emulator"]["traps_enabled"] = False
    elif change == "tools":
        report["live"]["configs"]["agent_config@2"]["tool_subset"] = []
    with pytest.raises(ValueError, match=message):
        remote.prepare_remote(None, cases, report, catalog)


def test_probe_is_self_contained_and_uses_shared_validators(monkeypatch, tmp_path):
    source = remote.probe_source()
    # Avoid the entry point: evaluate all definitions on an old-server-like
    # namespace, then exercise the actual shared catalog checker.
    definitions = source.rsplit("\nprint(json.dumps(probe(", 1)[0]
    namespace = {}
    exec(compile(definitions, "remote-probe", "exec"), namespace)
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "one.yaml").write_text(
        "name: one\ntitle: One\nkind: recipe\ndomain: org\ndescription: One\n"
        "query: {schema: dm_core, logic_model: employee_actual, metrics: [fact_count]}\n"
    )
    from heimdall.catalog.model import Catalog
    catalog = Catalog.load(Path(__file__).resolve().parents[1] / "catalog/snapshot.json")
    registry = namespace["validate_skill_catalog"](skills, catalog)
    assert registry.all_names() == ["one"]
    assert namespace["catalog_hash"](skills).startswith("sha256:")
    assert 'from sim.skill_eval.pin import pin' in source
    assert 'registry.commit(' not in source
    assert 'truth/' not in source
    assert "heimdall-emulator:8081" not in source
    assert "/app/heimdall-skills" not in source
    assert "/app/catalog/snapshot.json" not in source
    assert "require_env" in source
    assert "from sim.benchmark" not in source
    assert "sim.benchmark.env" not in source
    assert "127.0.0.1:8081" not in source
    assert "127.0.0.1:6006" not in source
    assert '_required_option(options, "control_url")' in source
    assert '_required_option(options, "phoenix_project")' in source


def test_ssh_sends_only_stdin_script_no_install_or_shell_interpolation(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout='{"ok": true}')
    monkeypatch.setattr(remote.subprocess, "run", run)
    assert remote.read_container("user@server", "admin", "registry", {"config_ref": "agent@2"}) == {"ok": True}
    argv, kwargs = calls[0]
    assert argv[:4] == ["ssh", "-o", "ConnectTimeout=10", "user@server"]
    assert "docker exec -i admin python -B - registry" in argv[-1]
    assert kwargs["check"] is True
    assert "input" in kwargs
    with pytest.raises(ValueError):
        remote.read_container("-oProxyCommand=bad", "admin", "registry", {})


@pytest.mark.parametrize("check_only", [False, True])
def test_remote_entrypoint_one_case_and_saved_results(monkeypatch, tmp_path, check_only):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    raw = json.loads(EXAMPLE.read_text())
    for case_id in ("case-0002", "case-0001"):
        (cases_dir / f"{case_id}.json").write_text(json.dumps({**raw, "case_id": case_id}))
    selected = cli.load_cases_path(cases_dir, limit=1)
    report, catalog = reports(selected[0])
    class Health:
        def get(self, url):
            assert url == "/agent/healthz"
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"status": "ok"})
        def close(self):
            pass
    class Stand:
        def __init__(self, **kwargs):
            self._client = Health()
        def phoenix_http(self):
            return self._client
    monkeypatch.setattr(remote, "StandClient", Stand)
    def read_container(ssh, container, kind, options, **kwargs):
        if kind == "registry":
            assert options["phoenix_project"] == "b2e-itmo"
            return deepcopy(report)
        assert options["control_url"] == "http://127.0.0.1:8081"
        return deepcopy(catalog)

    monkeypatch.setattr(remote, "read_container", read_container)
    monkeypatch.setattr(remote, "close_ssh_master", lambda *args: None)
    requests = []
    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["trace_backend"] == "phoenix"
        def execute(self, request):
            requests.append(request)
            prepared = remote.prepare_remote(None, selected, deepcopy(report), deepcopy(catalog))
            mode = next(
                name for name, ref in report["live"]["refs"].items()
                if ref == request.config_ref
            )
            fingerprint = prepared.checked[(selected[0].case_id, mode)].fingerprint.as_dict()
            return AgentTurn(
                json.dumps({"result": raw["evaluation_contract"]["gold_result"], "message": None}),
                stats={"total_tokens": 10}, trace={"spans": []}, fingerprint=fingerprint,
            )
    monkeypatch.setattr(remote, "StandSessionExecutor", Executor)
    output = tmp_path / "results"
    args = ["--cases", str(cases_dir), "--results", str(output), "--eval-id", "test"]
    assert remote.main(args + (["--check-only"] if check_only else [])) == 0
    assert len(requests) == (0 if check_only else 2)
    manifest = json.loads((output / "test/run-manifest.json").read_text())
    assert manifest["case_ids"] == ["case-0001"]
    assert manifest["live_stand"]["remote_catalog_validation"]["validated"] is True
    if not check_only:
        assert {request.config_ref for request in requests} == {
            "benchmark_general@1", "agent_config@2",
        }
        assert "headcount_by_dimension" not in requests[0].query
        assert "gold_result" not in requests[0].query
        scores = [
            json.loads(line)
            for line in (output / "test/scores.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(scores) == 2
        assert {row["metrics"]["answer_accuracy"] for row in scores} == {1}


def test_remote_refuses_generated_skills_mock():
    args = remote.parser().parse_args(["--cases", str(EXAMPLE), "--modes", "generated_skills"])
    with pytest.raises(ValueError, match="still a mock"):
        remote.selected_mode_names(args)


def test_make_remote_uses_server_runner_without_local_case_limit():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["make", "-n", "benchmark-remote", "CASES=some/cases"],
                            cwd=root, text=True, capture_output=True, check=True)
    assert "sim.benchmark.remote_server" in result.stdout
    assert "--limit" not in result.stdout
    assert '--repetitions "1"' in result.stdout
    assert "docker compose" not in result.stdout


def test_make_remote_smoke_keeps_local_driver_and_one_case():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-n", "benchmark-remote-smoke", "CASES=some/cases"],
        cwd=root, text=True, capture_output=True, check=True,
    )
    assert "sim.benchmark.remote --cases" in result.stdout
    assert '--limit "1"' in result.stdout
    assert '--emulator-control-url "http://127.0.0.1:8081"' in result.stdout
    assert '--phoenix-control-url "http://127.0.0.1:6006"' in result.stdout
    assert '--phoenix-project "b2e-itmo"' in result.stdout
    assert "docker compose" not in result.stdout


def test_make_server_runner_uses_internal_agent_and_phoenix():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-n", "benchmark-server", "CASES=benchmarking/cases"],
        cwd=root, text=True, capture_output=True, check=True,
    )
    assert "benchmark-runner" in result.stdout
    assert "--agent-url http://b2e-agent:8082" in result.stdout
    assert "--phoenix-url http://phoenix:6006" in result.stdout
    assert "--no-auth" in result.stdout


def test_remote_phoenix_trace_uses_internal_container_api(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=json.dumps({"spans": [{"name": "b2e.turn"}]}))

    monkeypatch.setattr(remote.subprocess, "run", run)
    spans = remote.read_phoenix_trace(
        "user@server",
        "phoenix-container",
        "b2e-itmo",
        "trace-1",
        control_url="http://127.0.0.1:6006",
        control_path="/tmp/control",
    )

    assert spans == [{"name": "b2e.turn"}]
    argv, kwargs = calls[0]
    assert argv[-2] == "user@server"
    assert (
        "docker exec -i phoenix-container python -B - "
        "b2e-itmo trace-1 http://127.0.0.1:6006"
    ) == argv[-1]
    assert "127.0.0.1:6006" not in kwargs["input"]
    assert "ControlPath=/tmp/control" in argv


def test_remote_phoenix_trace_rejects_malformed_payload(monkeypatch):
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"unexpected": true}'),
    )
    with pytest.raises(ValueError, match="no spans array"):
        remote.read_phoenix_trace(
            "user@server",
            "phoenix",
            "b2e-itmo",
            "trace-1",
            control_url="http://127.0.0.1:6006",
        )
