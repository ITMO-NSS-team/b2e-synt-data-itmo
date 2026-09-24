"""Make targets for running the benchmark beside an existing server stand."""
from __future__ import annotations

from pathlib import Path
import subprocess


def _make(*arguments: str) -> str:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-n", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_server_runner_uses_internal_agent_and_phoenix() -> None:
    output = _make("benchmark-server", "CASES=benchmarking/cases")

    assert "benchmark-runner" in output
    assert "--agent-url http://b2e-agent:8082" in output
    assert "--phoenix-url http://phoenix:6006" in output
    assert "--no-auth" in output
    assert "ssh" not in output


def test_server_smoke_runs_one_case_and_one_repetition() -> None:
    output = _make("benchmark-server-smoke", "CASES=benchmarking/cases")

    assert 'repetitions "1"' in output
    assert '--limit "1"' in output
    assert "benchmark-runner" in output


def test_remote_targets_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    for target in ("benchmark-remote", "benchmark-remote-smoke"):
        result = subprocess.run(
            ["make", "-n", target],
            cwd=root,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
