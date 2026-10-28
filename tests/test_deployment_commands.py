"""Exercise Make against a fake Docker CLI; never touch the running stand."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def project(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "Makefile", tmp_path / "Makefile")
    (tmp_path / "deploy").mkdir()
    (tmp_path / "sim/benchmark").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text("{}")
    (tmp_path / "generated-skills").mkdir()
    (tmp_path / "deploy/.env").write_text(
        "B2E_HARNESS=messages_api\nB2E_MODEL=test-model\n"
        "BENCH_MODES=wrong-deploy-setting\n"
    )
    (tmp_path / "sim/benchmark/.env").write_text(
        "BENCHMARK_REPO_DIR=basket\n"
        "GENERATED_SKILLS_DIR=generated-skills\n"
        "CASES=basket/gold_dataset\n"
        "BENCH_MODES=general_knowledge,skills_disabled,existing_skills,generated_skills\n"
        "BENCH_REPETITIONS=2\n"
    )
    docker = tmp_path / "fake-docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "with pathlib.Path('docker-calls.jsonl').open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if 'ps' in sys.argv and pathlib.Path('invalid-compose').exists():\n"
        "    print('invalid compose project', file=sys.stderr)\n"
        "    sys.exit(7)\n"
    )
    docker.chmod(0o755)
    return tmp_path, docker


def run_make(project, *args: str, input: str | None = None):
    root, docker = project
    return subprocess.run(
        ["make", f"DOCKER={docker}", f"RESOLV_OVERLAY={root}/resolv.conf", f"RESOLV_RECREATE_MARKER={root}/var/resolv-recreate", *args],
        cwd=root, text=True, capture_output=True, input=input,
    )


def calls(project):
    root, _ = project
    return [json.loads(line) for line in (root / "docker-calls.jsonl").read_text().splitlines()]


def test_up_retains_original_build_and_dns_recreation_behavior(project):
    root, _ = project
    result = run_make(project, "up")
    assert result.returncode == 0, result.stderr
    invocation, = calls(project)
    assert "--build" in invocation
    assert "--force-recreate" in invocation
    assert "--no-build" not in invocation
    assert "nameserver 127.0.0.11" in (root / "resolv.conf").read_text()
    assert not (root / "var/resolv-recreate").exists()


def test_up_does_not_force_recreate_when_dns_overlay_already_exists(project):
    root, _ = project
    (root / "resolv.conf").write_text("nameserver 127.0.0.11\n")
    result = run_make(project, "up")
    assert result.returncode == 0, result.stderr
    invocation, = calls(project)
    assert "--build" in invocation and "--force-recreate" not in invocation


def test_relative_env_paths_and_mode_defaults_are_resolved(project):
    root, _ = project
    probe = (
        "probe:\n\n"
        "$(info MODES=$(BENCH_MODES))\n"
        "$(info REPETITIONS=$(BENCH_REPETITIONS))\n"
        "$(info SKILLS=$(GENERATED_SKILLS_DIR))\n"
        "$(info CASES=$(BENCH_CASES_HOST))\n"
        "$(info CONTAINER_CASES=$(BENCH_CASES_CONTAINER))\n"
        "$(info BENCH_COMPOSE=$(BENCH_COMPOSE))\n"
    )
    result = run_make(project, "-f", "Makefile", "-f", "-", "probe", input=probe)
    assert result.returncode == 0, result.stderr
    assert "MODES=general_knowledge,skills_disabled,existing_skills,generated_skills" in result.stdout
    assert "REPETITIONS=2" in result.stdout
    assert f"SKILLS={root}/generated-skills" in result.stdout
    assert f"CASES={root}/basket/gold_dataset" in result.stdout
    assert "CONTAINER_CASES=/app/b2e-skill-benchmark/gold_dataset" in result.stdout
    assert f'BENCHMARK_REPO_DIR="{root}/basket"' in result.stdout
    result = run_make(project, "-f", "Makefile", "-f", "-", "probe", "BENCH_MODES=skills_disabled", input=probe)
    assert "MODES=skills_disabled" in result.stdout


def test_compose_error_does_not_start_or_build_another_stand(project):
    root, _ = project
    (root / "invalid-compose").touch()
    result = run_make(project, "_benchmarking-single", "BENCH_MODES=skills_disabled")
    assert result.returncode != 0
    assert "invalid compose project" in result.stderr
    assert "Compose inspection failed" in result.stdout
    assert len(calls(project)) == 1
    assert "ps" in calls(project)[0]


def test_absent_mode_setting_retains_all_three_baselines(project):
    root, _ = project
    env = root / "sim/benchmark/.env"
    env.write_text("")
    probe = "probe:\n\n$(info MODES=$(BENCH_MODES))\n"
    result = run_make(project, "-f", "Makefile", "-f", "-", "probe", input=probe)
    assert result.returncode == 0, result.stderr
    assert "MODES=general_knowledge,skills_disabled,existing_skills" in result.stdout


def test_make_case_override_remains_relative_to_repo(project):
    root, _ = project
    probe = "probe:\n\n$(info CASES=$(BENCH_CASES_HOST))\n$(info CONTAINER=$(BENCH_CASES_CONTAINER))\n"
    result = run_make(project, "-f", "Makefile", "-f", "-", "probe", "CASES=var/staged.jsonl", input=probe)
    assert result.returncode == 0, result.stderr
    assert f"CASES={root}/var/staged.jsonl" in result.stdout
    assert "CONTAINER=/app/var/staged.jsonl" in result.stdout
