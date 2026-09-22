"""Server-side remote orchestration stays shell-safe and read-only locally."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.benchmark import remote_server


def test_server_make_command_is_deterministic_and_quoted() -> None:
    args = remote_server.parser().parse_args([
        "--cases", "benchmarking/cases",
        "--remote-root", "/srv/b2e stand",
        "--eval-id", "evaluation-1",
        "--limit", "3",
    ])

    command, eval_id = remote_server.server_make_command(args)

    assert eval_id == "evaluation-1"
    assert command.startswith("cd '/srv/b2e stand' && make benchmark-server")
    assert "CASES=benchmarking/cases" in command
    assert "BENCH_EVAL_ID=evaluation-1" in command
    assert "BENCH_LIMIT=3" in command


@pytest.mark.parametrize("value", ["/absolute/cases", "../cases", "cases/../../x"])
def test_server_make_command_rejects_cases_outside_checkout(value: str) -> None:
    args = remote_server.parser().parse_args(["--cases", value])
    with pytest.raises(ValueError, match="inside the server checkout"):
        remote_server.server_make_command(args)


def test_main_runs_one_remote_job_then_downloads_results(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_server.subprocess, "run", run)
    monkeypatch.setattr(remote_server, "close_ssh_master", lambda *args: None)

    code = remote_server.main([
        "--cases", "benchmarking/cases",
        "--ssh", "user@server",
        "--remote-root", "/srv/repo",
        "--local-results", str(tmp_path),
        "--eval-id", "evaluation-1",
    ])

    assert code == 0
    assert len(calls) == 2
    ssh, scp = calls
    assert ssh[0][-2] == "user@server"
    assert "make benchmark-server" in ssh[0][-1]
    assert scp[0][0:2] == ["scp", "-r"]
    assert "user@server:/srv/repo/benchmarking/results/evaluation-1" in scp[0]


def test_main_refuses_to_overwrite_local_results(tmp_path: Path) -> None:
    (tmp_path / "evaluation-1").mkdir()
    code = remote_server.main([
        "--cases", "benchmarking/cases",
        "--local-results", str(tmp_path),
        "--eval-id", "evaluation-1",
    ])
    assert code == 2
