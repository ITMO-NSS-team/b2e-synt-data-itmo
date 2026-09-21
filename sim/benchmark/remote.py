"""Local benchmark against an unchanged remote stand and its current model.

SSH reads settings and validates catalogs in place; only metadata is returned.
Sessions and traces use HTTPS. No server config writes, Compose or migrations.
"""
from __future__ import annotations

import inspect
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import httpx

from sim.fingerprint import RunFingerprint
from sim.skill_eval.stand import StandClient

from . import cli
from .catalog_snapshots import verify_snapshot
from .execution import PinnedConfigActivator
from .modes import ModeConfig, SKILL_TOOLS, _loaded_registry, _skill_files, catalog_hash
from .preflight import PreflightResult, _check_query, validate_skill_catalog
from .path_lib import SCHEMA_RELATIVE


def probe_source() -> str:
    """Reuse the exact local validators on an older server via Python stdin."""
    source = Path(__file__).with_name("remote_probe.py").read_text(encoding="utf-8")
    functions = (_skill_files, catalog_hash, _loaded_registry, verify_snapshot,
                 _check_query, validate_skill_catalog)
    source += "\n" + "\n\n".join(inspect.getsource(func) for func in functions)
    return source + '\nprint(json.dumps(probe(sys.argv[1], json.loads(sys.argv[2])), ensure_ascii=False))\n'


def read_container(ssh: str, container: str, kind: str, options: dict) -> dict:
    """Execute a read-only probe, without installing files on the server."""
    if not ssh or ssh.startswith("-"):
        raise ValueError("--ssh must be a hostname or user@hostname")
    command = shlex.join([
        "docker", "exec", "-i", container, "python", "-B", "-", kind, json.dumps(options),
    ])
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", ssh, command],
        input=probe_source(), text=True, stdout=subprocess.PIPE, check=True, timeout=120,
    )
    return json.loads(result.stdout)


def prepare_remote(args, cases, report: dict, catalog: dict) -> cli.PreparedBenchmark:
    """Check public contracts, live conditions, validation report and scopes."""
    live = report["live"]
    common = cli.common_conditions(live, ("existing_skills",))
    for key in ("data_snapshot_hash", "traps_enabled", "latency_profile", "hr_employee_ids"):
        if live["emulator"][key] != catalog["emulator"][key]:
            raise ValueError(f"remote conditions changed during discovery: {key}")
    if catalog["snapshot_id"] != common.snapshot_id:
        raise ValueError("remote snapshot manifest differs from live snapshot")
    if (catalog.get("validated") is not True or catalog.get("skill_count", 0) < 1
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", catalog.get("catalog_hash", ""))):
        raise ValueError("remote skill catalog was not validated")
    # No local copy: remote location/hash and validation proof stay in manifest.
    mode = ModeConfig("existing_skills", SKILL_TOOLS, None, catalog["catalog_hash"],
                      None, None, (), common)
    ref = live["refs"][mode.name]
    activator = PinnedConfigActivator(
        {mode.name: ref}, config_reader=lambda name: live["configs"][name],
    )
    activator.activate(mode)
    checked = {}
    for case in cases:
        for field, expected in (("snapshot_id", common.snapshot_id),
                                ("skill_registry_hash", live["skill_registry_hash"])):
            if case.raw[field] != expected:
                raise ValueError(f"{case.case_id}: {field} differs from remote stand "
                                 f"(case={case.raw[field]}, stand={expected}); case was not changed")
        scope = report["scopes"][str(case.raw["employee_id"])]
        if (str(scope["employee_id"]) != str(case.raw["employee_id"])
                or scope["role"] != case.raw["employee_role"]):
            raise ValueError(f"{case.case_id}: remote employee role/identity mismatch")
        fingerprint = RunFingerprint.create(
            agent_config_version=ref, prompt_registry_version=common.prompt_registry_version,
            skill_registry_hash=live["skill_registry_hash"], model_id=common.model_id,
            temperature=common.temperature, data_snapshot_hash=common.snapshot_id,
            traps_enabled=common.traps_enabled, latency_profile=common.latency_profile,
            hr_employee_ids=common.hr_employee_ids,
        )
        checked[(case.case_id, mode.name)] = PreflightResult(case.case_id, mode.name, "ready", fingerprint)
    live["remote_catalog_validation"] = catalog
    return cli.PreparedBenchmark(cases, live, {mode.name: mode}, activator, checked)


def parser():
    result = argparse.ArgumentParser(
        description="Run locally against the unchanged remote agent, with its existing model/skills.")
    result.add_argument("--cases", required=True, help="Directory, JSON case, or JSONL suite")
    result.add_argument("--schema", default=SCHEMA_RELATIVE)
    result.add_argument("--results", default="benchmarking/results")
    result.add_argument("--limit", type=int, default=1, help="Number of ready cases (default: 1)")
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--modes", default="existing_skills", help="Only existing_skills is supported")
    result.add_argument("--eval-id")
    result.add_argument("--eval-prefix", default="remote")
    result.add_argument("--check-only", action="store_true")
    result.add_argument("--env-file", default="deploy/.env.remote")
    result.add_argument("--timeout", type=float, default=1800.0)
    result.add_argument("--trace-attempts", type=int, default=30)
    result.set_defaults(trace_backend="phoenix")
    result.add_argument("--ssh", default="nnikitin@10.32.1.71")
    result.add_argument("--public-url", default="https://10.32.1.71:8443")
    result.add_argument("--admin-container", default="b2e-itmo-admin-ui-1")
    result.add_argument("--emulator-container", default="b2e-itmo-heimdall-emulator-1")
    result.add_argument("--config-ref", default="agent_config")
    return result


def main(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
        if args.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if args.modes != "existing_skills":
            raise ValueError("remote run supports only existing_skills without modifying server configs")
        cases = cli.load_cases_path(args.cases, schema_path=args.schema, limit=args.limit)
        stand = StandClient(env_file=args.env_file, public_url=args.public_url, timeout=30)
        try:
            response = stand._client.get("/agent/healthz")
            response.raise_for_status()
            if response.json().get("status") != "ok":
                raise ValueError("remote agent health is not ok")
        finally:
            stand._client.close()
            stand.phoenix_http().close()
        print("Reading remote conditions and validating catalogs in place (no LLM calls)...", flush=True)
        report = read_container(args.ssh, args.admin_container, "registry", {
            "config_ref": args.config_ref,
            "employee_ids": sorted({str(case.raw["employee_id"]) for case in cases}),
        })
        catalog = read_container(args.ssh, args.emulator_container, "catalog", {})
        prepared = prepare_remote(args, cases, report, catalog)
        if args.check_only:
            from .results import ResultWriter
            writer = ResultWriter(args.results, cli._eval_id(args, "remote-check"))
            writer.write_manifest(cli._manifest(args, prepared, writer.root.name, check_only=True))
            print(json.dumps({"preflight": "ok", "agent_calls": 0, "results_dir": str(writer.root)}, indent=2))
            return 0
        print(f"Running {len(cases)} case(s) × {args.repetitions} repeat(s), existing_skills, "
              f"model={next(iter(prepared.selected_modes.values())).common.model_id}. "
              "This uses the server's paid API account.", flush=True)
        results, writer = cli.run(args, prepared=prepared)
        failed = sum(result.status != "completed" for result in results)
        print(json.dumps({"results_dir": str(writer.root), "runs": len(results),
                          "pipeline_failures": failed, "summary": str(writer.root / "summary.json")},
                         ensure_ascii=False, indent=2))
        return 2 if failed else 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, httpx.HTTPError) as exc:
        print(f"remote benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
