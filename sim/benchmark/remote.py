"""Temporary local smoke/debug driver against a remote stand.

SSH pins append-only ``general_knowledge`` / ``existing_skills`` clones of the
live ``agent_config`` and validates catalogs in place. Sessions and traces use
HTTPS. The live ``agent_config`` head, Compose and the skill catalog are not
rewritten. ``generated_skills`` stays a mock. Full experiments use
``sim.benchmark.remote_server`` and execute beside the stand instead.
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
import tempfile

import httpx

from sim.fingerprint import RunFingerprint
from sim.skill_eval.stand import StandClient

from . import cli
from .catalog_snapshots import verify_snapshot
from .env import load_env, require_env
from .execution import PinnedConfigActivator, StandSessionExecutor
from .modes import (
    ModeConfig, _loaded_registry, _skill_files, catalog_hash, mode_strategies,
    mode_strategy,
)
from .preflight import PreflightResult, _check_query, validate_skill_catalog
from .path_lib import SCHEMA_RELATIVE


def probe_source() -> str:
    """Reuse the exact local validators on an older server via Python stdin."""
    source = Path(__file__).with_name("remote_probe.py").read_text(encoding="utf-8")
    functions = (
        load_env, require_env, _skill_files, catalog_hash, _loaded_registry,
        verify_snapshot, _check_query, validate_skill_catalog,
    )
    source += "\nENV_PATH = Path('deploy/.env')\n"
    source += "\n" + "\n\n".join(inspect.getsource(func) for func in functions)
    return source + '\nprint(json.dumps(probe(sys.argv[1], json.loads(sys.argv[2])), ensure_ascii=False))\n'


def ssh_command(ssh: str, control_path: str | None, command: str) -> list[str]:
    result = ["ssh", "-o", "ConnectTimeout=10"]
    if control_path:
        result.extend([
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control_path}",
            "-o", "ControlPersist=60",
        ])
    return [*result, ssh, command]


def read_container(
    ssh: str, container: str, kind: str, options: dict,
    *, control_path: str | None = None,
) -> dict:
    """Execute a probe on the server without installing files.

    The registry probe append-only pins the two benchmark agent configs.
    """
    if not ssh or ssh.startswith("-"):
        raise ValueError("--ssh must be a hostname or user@hostname")
    command = shlex.join([
        "docker", "exec", "-i", container, "python", "-B", "-", kind, json.dumps(options),
    ])
    result = subprocess.run(
        ssh_command(ssh, control_path, command),
        input=probe_source(), text=True, stdout=subprocess.PIPE, check=True, timeout=120,
    )
    return json.loads(result.stdout)


_PHOENIX_TRACE_SOURCE = r'''import json
import sys
import urllib.parse
import urllib.request

project, trace_id, base = sys.argv[1:4]
endpoint = base.rstrip("/") + "/v1/projects/" + urllib.parse.quote(project, safe="") + "/spans"
spans = {}
cursor = None
seen_cursors = set()
scanned = 0
while True:
    params = {"trace_id": trace_id, "limit": 1000}
    if cursor:
        params["cursor"] = cursor
    with urllib.request.urlopen(endpoint + "?" + urllib.parse.urlencode(params), timeout=30) as response:
        content_type = response.headers.get_content_type()
        if response.status != 200 or content_type != "application/json":
            raise RuntimeError(f"Phoenix returned {response.status} {content_type}")
        body = json.load(response)
    batch = body.get("data", []) if isinstance(body, dict) else []
    scanned += len(batch)
    if scanned > 20000:
        raise RuntimeError("Phoenix trace exceeded 20000 scanned spans")
    for span in batch:
        context = span.get("context") or {}
        if str(context.get("trace_id") or span.get("trace_id") or "") != trace_id:
            continue
        span_id = context.get("span_id") or span.get("span_id")
        if not span_id:
            raise RuntimeError("Phoenix span has no span_id")
        spans[str(span_id)] = span
    cursor = body.get("next_cursor") if isinstance(body, dict) else None
    if not cursor:
        break
    if cursor in seen_cursors:
        raise RuntimeError("Phoenix pagination cursor repeated")
    seen_cursors.add(cursor)
print(json.dumps({"spans": list(spans.values())}, ensure_ascii=False))
'''


def read_phoenix_trace(
    ssh: str,
    container: str,
    project: str,
    trace_id: str,
    *,
    control_url: str,
    control_path: str | None = None,
) -> list[dict]:
    """Read one trace from Phoenix's internal REST API over read-only SSH."""
    if not ssh or ssh.startswith("-"):
        raise ValueError("--ssh must be a hostname or user@hostname")
    base = control_url.strip()
    if not base:
        raise ValueError("phoenix probe requires control_url")
    command = shlex.join([
        "docker", "exec", "-i", container, "python", "-B", "-",
        project, trace_id, base,
    ])
    result = subprocess.run(
        ssh_command(ssh, control_path, command),
        input=_PHOENIX_TRACE_SOURCE,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        timeout=120,
    )
    payload = json.loads(result.stdout)
    spans = payload.get("spans") if isinstance(payload, dict) else None
    if not isinstance(spans, list):
        raise ValueError("remote Phoenix probe returned no spans array")
    return spans


def close_ssh_master(ssh: str, control_path: str) -> None:
    """Close a multiplexed SSH connection; ignore an already-closed master."""
    subprocess.run(
        ["ssh", "-S", control_path, "-O", "exit", ssh],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=15,
    )


def selected_mode_names(args=None) -> tuple[str, ...]:
    """Parse remote ``--modes``; generated_skills stays a refused mock."""
    raw = cli.DEFAULT_MODES if args is None else args.modes
    if not isinstance(raw, str):
        names = tuple(raw)
    else:
        names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        raise ValueError("at least one benchmark mode is required")
    if len(names) != len(set(names)):
        raise ValueError("benchmark modes contain duplicates")
    allowed = {strategy.name for strategy in mode_strategies() if strategy.remote_supported}
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(
            "remote run supports general_knowledge and existing_skills; "
            "generated_skills is still a mock: " + ", ".join(sorted(unknown))
        )
    return names


def _remote_mode(name: str, common, catalog_hash: str) -> ModeConfig:
    strategy = mode_strategy(name)
    return ModeConfig(
        name,
        strategy.tool_subset,
        None,
        catalog_hash if strategy.requires_catalog else None,
        None,
        None,
        (),
        common,
        skills_enabled=strategy.skills_enabled,
    )


def prepare_remote(args, cases, report: dict, catalog: dict) -> cli.PreparedBenchmark:
    """Check public contracts, live conditions, validation report and scopes."""
    live = report["live"]
    names = selected_mode_names(args)
    common = cli.common_conditions(live, names)
    for key in ("data_snapshot_hash", "traps_enabled", "latency_profile", "hr_employee_ids"):
        if live["emulator"][key] != catalog["emulator"][key]:
            raise ValueError(f"remote conditions changed during discovery: {key}")
    if catalog["snapshot_id"] != common.snapshot_id:
        raise ValueError("remote snapshot manifest differs from live snapshot")
    if (catalog.get("validated") is not True or catalog.get("skill_count", 0) < 1
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", catalog.get("catalog_hash", ""))):
        raise ValueError("remote skill catalog was not validated")
    selected = {name: _remote_mode(name, common, catalog["catalog_hash"]) for name in names}
    activator = PinnedConfigActivator(
        {name: live["refs"][name] for name in selected},
        config_reader=lambda name: live["configs"][name],
    )
    for mode in selected.values():
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
        for mode in selected.values():
            fingerprint = RunFingerprint.create(
                agent_config_version=live["refs"][mode.name],
                prompt_registry_version=common.prompt_registry_version,
                skill_registry_hash=live["skill_registry_hash"],
                model_id=common.model_id,
                temperature=common.temperature,
                data_snapshot_hash=common.snapshot_id,
                traps_enabled=common.traps_enabled,
                latency_profile=common.latency_profile,
                hr_employee_ids=common.hr_employee_ids,
            )
            checked[(case.case_id, mode.name)] = PreflightResult(
                case.case_id, mode.name, "ready", fingerprint,
            )
    live["remote_catalog_validation"] = catalog
    return cli.PreparedBenchmark(cases, live, selected, activator, checked)


def parser():
    result = argparse.ArgumentParser(
        description="Run locally against the remote agent: general_knowledge and existing_skills.")
    result.add_argument("--cases", required=True, help="Directory, JSON case, or JSONL suite")
    result.add_argument("--schema", default=SCHEMA_RELATIVE)
    result.add_argument("--results", default="benchmarking/results")
    result.add_argument("--limit", type=int, default=1, help="Number of ready cases (default: 1)")
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument(
        "--modes", default=",".join(cli.DEFAULT_MODES),
        help="general_knowledge,existing_skills; generated_skills is still a mock",
    )
    result.add_argument("--eval-id")
    result.add_argument("--eval-prefix", default="remote")
    result.add_argument("--check-only", action="store_true")
    result.add_argument("--env-file", default="deploy/.env")
    result.add_argument("--timeout", type=float, default=1800.0)
    result.add_argument(
        "--trace-timeout", type=float, default=300.0,
        help="Maximum seconds to wait for a complete trace",
    )
    result.set_defaults(trace_backend="phoenix")
    result.add_argument("--ssh", default="nnikitin@10.32.1.71")
    result.add_argument("--public-url", default="https://10.32.1.71:8443")
    result.add_argument("--admin-container", default="b2e-itmo-admin-ui-1")
    result.add_argument("--emulator-container", default="b2e-itmo-heimdall-emulator-1")
    result.add_argument(
        "--emulator-control-url", default="http://127.0.0.1:8081",
        help="Emulator control API as seen from inside --emulator-container",
    )
    result.add_argument("--phoenix-container", default="b2e-itmo-phoenix-1")
    result.add_argument(
        "--phoenix-control-url", default="http://127.0.0.1:6006",
        help="Phoenix REST API as seen from inside --phoenix-container",
    )
    result.add_argument(
        "--phoenix-project", default="b2e-itmo",
        help="Phoenix project name inside the stand",
    )
    result.add_argument("--config-ref", default="agent_config")
    return result


def main(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
        if args.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        selected_mode_names(args)
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
        with tempfile.TemporaryDirectory(prefix="b2e-bench-", dir="/tmp") as ssh_dir:
            control_path = str(Path(ssh_dir) / "ssh")
            try:
                print("Pinning benchmark configs and validating catalogs in place (no LLM calls)...", flush=True)
                report = read_container(args.ssh, args.admin_container, "registry", {
                    "config_ref": args.config_ref,
                    "employee_ids": sorted({str(case.raw["employee_id"]) for case in cases}),
                    "phoenix_project": args.phoenix_project,
                }, control_path=control_path)
                catalog = read_container(
                    args.ssh, args.emulator_container, "catalog",
                    {"control_url": args.emulator_control_url},
                    control_path=control_path,
                )
                prepared = prepare_remote(args, cases, report, catalog)
                if args.check_only:
                    from .results import ResultWriter
                    writer = ResultWriter(args.results, cli._eval_id(args, "remote-check"))
                    writer.write_manifest(cli._manifest(args, prepared, writer.root.name, check_only=True))
                    print(json.dumps({"preflight": "ok", "agent_calls": 0, "results_dir": str(writer.root)}, indent=2))
                    return 0
                print(
                    f"Running {len(cases)} case(s) × {args.repetitions} repeat(s), "
                    f"{', '.join(prepared.selected_modes)}, "
                    f"model={next(iter(prepared.selected_modes.values())).common.model_id}. "
                    "This uses the server's paid API account.",
                    flush=True,
                )
                executor = StandSessionExecutor(
                    env_file=args.env_file,
                    public_url=args.public_url,
                    timeout=args.timeout,
                    trace_timeout=args.trace_timeout,
                    trace_backend="phoenix",
                    phoenix_trace_fetcher=lambda trace_id: read_phoenix_trace(
                        args.ssh,
                        args.phoenix_container,
                        report["live"]["phoenix_project"],
                        trace_id,
                        control_url=args.phoenix_control_url,
                        control_path=control_path,
                    ),
                )
                results, writer = cli.run(args, prepared=prepared, executor=executor)
                print(json.dumps(cli._finished_report(results, writer), ensure_ascii=False, indent=2))
                return 0
            finally:
                close_ssh_master(args.ssh, control_path)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, httpx.HTTPError) as exc:
        print(f"remote benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
