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
    assert "prepare-resolv" in output
    assert "openlit-dashboard" in output
    assert '--modes "general_knowledge,skills_disabled,existing_skills"' in output
    assert 'HEIMDALL_SKILLS_ROOT="/app/heimdall-skill-catalog/existing"' in output
    assert "ssh" not in output


def test_prepare_resolv_repairs_docker_created_directory() -> None:
    output = _make("prepare-resolv")

    assert "rmdir" in output
    assert ".resolv-recreate-required" in output
    assert "nameserver 127.0.0.11" in output
    assert "chmod 644" in output


def test_up_force_recreates_services_after_resolv_repair() -> None:
    output = _make("up")

    assert ".resolv-recreate-required" in output
    assert '--force-recreate' in output


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
    assert "openlit-dashboard" not in output


def test_generated_benchmark_mounts_only_generated_catalog(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "skill.md").write_text("placeholder", encoding="utf-8")

    output = _make(
        "benchmarking-generated",
        "CASES=benchmarking/cases",
        f"GENERATED_SKILLS_DIR={generated}",
    )

    assert 'HEIMDALL_SKILLS_ROOT="/app/heimdall-skill-catalog/generated"' in output
    assert '--modes "generated_skills"' in output


def test_mixed_benchmark_splits_catalogs_and_merges_results() -> None:
    # GNU make executes recursive $(MAKE) calls even under -n, so this dispatch
    # recipe is asserted structurally; its merge behavior has dedicated tests.
    root = Path(__file__).resolve().parents[1]
    source = (root / "Makefile").read_text(encoding="utf-8")

    assert "Фаза 1/2" in source
    assert "Фаза 2/2" in source
    assert 'BENCH_MODES="$(BENCH_BASELINE_MODES)"' in source
    assert 'BENCH_MODES="generated_skills"' in source
    assert "sim.benchmark.merge" in source


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
