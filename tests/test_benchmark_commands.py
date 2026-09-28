"""Make targets for the single in-network benchmark path."""
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


def test_benchmarking_uses_internal_agent_and_phoenix() -> None:
    output = _make("benchmarking", "CASES=benchmarking/cases")

    assert "benchmark-runner" in output
    assert "--agent-url http://b2e-agent:8082" in output
    assert "--phoenix-url http://phoenix:6006" in output
    assert "--no-auth" in output
    assert "ps --status running --services" in output
    assert "make up" in output
    assert '--modes "general_knowledge,skills_disabled,existing_skills"' in output
    assert "ssh" not in output


def test_benchmarking_smoke_runs_one_case_and_one_repetition() -> None:
    output = _make("benchmarking-smoke", "CASES=benchmarking/cases")

    assert 'repetitions "1"' in output
    assert '--limit "1"' in output
    assert "benchmark-runner" in output


def test_benchmarking_check_is_preflight_only() -> None:
    output = _make("benchmarking-check", "CASES=benchmarking/cases")

    assert "--check-only" in output
    assert '--eval-prefix "check"' in output
    assert "benchmark-runner" in output


def test_data_dir_override_is_used_by_preflight() -> None:
    output = _make(
        "benchmarking-check",
        "CASES=benchmarking/cases",
        "DATA_DIR=/tmp/full-snapshot",
    )

    assert '/tmp/full-snapshot/manifest.json' in output


def test_smoke_uses_the_same_data_dir_as_the_stand() -> None:
    output = _make("smoke", "DATA_DIR=/tmp/full-snapshot")

    assert 'SMOKE_DATA="/tmp/full-snapshot"' in output


def test_old_targets_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    for target in (
        "benchmark-run", "benchmark-server", "benchmark-check",
        "benchmark-smoke", "benchmark-data-check", "benchmark-live-config",
        "eval-skills", "eval-deps", "pin-eval-configs",
    ):
        result = subprocess.run(
            ["make", "-n", target],
            cwd=root,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
