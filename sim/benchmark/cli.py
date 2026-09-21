"""Preflight and run benchmark cases against a real Compose stand."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .cases import BenchmarkCase, load_case, load_suite, validate_case
from .execution import PinnedConfigActivator, StandSessionExecutor
from .modes import BenchmarkMode, CommonConditions, ModeConfig, build_modes
from .path_lib import ENV_RELATIVE, SCHEMA_RELATIVE
from .preflight import PreflightResult, preflight_case
from .results import ResultWriter, summarize_results
from .runner import BenchmarkRunner

DEFAULT_MODES = (
    BenchmarkMode.SKILLS_DISABLED.value,
    BenchmarkMode.EXISTING_SKILLS.value,
)


@dataclass(frozen=True, slots=True)
class PreparedBenchmark:
    """Validated matrix ready to check or run.

    Attributes:
        cases: Ready cases selected from the CLI path.
        live: Secret-free stand manifest from ``live_config``.
        selected_modes: Arms requested on the command line.
        activator: Pinned-config activator bound to the live refs.
        checked: Preflight result per ``(case_id, mode)``.
    """
    cases: list[BenchmarkCase]
    live: dict[str, Any]
    selected_modes: dict[str, ModeConfig]
    activator: PinnedConfigActivator
    checked: dict[tuple[str, str], PreflightResult]


def load_cases_path(
    source: str | Path,
    *,
    schema_path: str | Path | None = None,
    limit: int | None = None,
) -> list[BenchmarkCase]:
    """Accept a case directory, one authorial JSON, or a derived JSONL suite.

    Args:
        source: Directory of ``*.json``, a single case file, or a ``.jsonl`` suite.
        schema_path: Optional authorial schema override.
        limit: If set, keep only the first N ready cases after sort.

    Returns:
        Ready cases in deterministic order.

    Raises:
        ValueError: If the path kind is unknown or contains no ready cases.
    """
    path = Path(source)
    if path.is_dir():
        cases = load_suite(path, on_draft="skip", schema_path=schema_path)
    elif path.is_file() and path.suffix == ".json":
        case = load_case(path, schema_path=schema_path)
        cases = [] if case.status == "draft" else [case]
    elif path.is_file() and path.suffix == ".jsonl":
        cases = _load_jsonl(path, schema_path=schema_path)
    else:
        raise ValueError(
            f"cases path must be a directory, .json or .jsonl file: {path}"
        )
    if not cases:
        raise ValueError(f"cases path contains no ready cases: {path}")
    if limit is not None:
        if limit < 1:
            raise ValueError("case limit must be >= 1")
        cases = cases[:limit]
    return cases


def _load_jsonl(
    path: Path, *, schema_path: str | Path | None,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        validate_case(raw, schema_path=schema_path)
        case = BenchmarkCase(path.resolve(), raw)
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id in {path}: {case.case_id}")
        seen.add(case.case_id)
        if case.status == "ready":
            cases.append(case)
    return sorted(cases, key=lambda item: item.case_id)


def load_live_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a secret-free stand manifest.

    Args:
        path: JSON written by ``python -m sim.benchmark.live_config``.

    Returns:
        Manifest with refs, configs, prompt versions and emulator condition.

    Raises:
        ValueError: If required fields are missing or the schema version differs.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "refs", "configs", "prompt_versions",
        "skill_registry_hash", "emulator",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("live stand manifest has missing or unknown fields")
    if payload["schema_version"] != "1.0":
        raise ValueError("unsupported live stand manifest version")
    for field in ("refs", "configs", "prompt_versions", "emulator"):
        if not isinstance(payload[field], dict):
            raise ValueError(f"live stand manifest {field} must be an object")
    return payload


def common_conditions(
    payload: dict[str, Any], mode_names: Iterable[str] = DEFAULT_MODES,
) -> CommonConditions:
    """Derive shared conditions from the selected pinned configurations.

    Args:
        payload: Validated live-stand manifest.
        mode_names: Configurations to compare; remote can use only existing_skills.

    Returns:
        CommonConditions copied into every mode.

    Raises:
        ValueError: If skills-on/off configs disagree on model, prompt or code policy.
    """
    refs = payload["refs"]
    configs = payload["configs"]
    prompts = payload["prompt_versions"]
    mode_names = tuple(mode_names)
    if not mode_names:
        raise ValueError("at least one mode is required for common conditions")
    required_modes = set(mode_names)
    if not required_modes <= refs.keys():
        raise ValueError("live stand manifest has no pinned skills-on/off refs")
    selected = [configs[refs[name]] for name in mode_names]
    stable_fields = (
        "model_id", "temperature", "code_execution", "conversation_mode",
    )
    for field in stable_fields:
        if len({json.dumps(config.get(field), sort_keys=True) for config in selected}) != 1:
            raise ValueError(f"pinned benchmark configs differ by {field}")
    prompt_refs = {prompts[refs[name]] for name in mode_names}
    if len(prompt_refs) != 1:
        raise ValueError("pinned benchmark configs resolve to different prompts")
    emulator = payload["emulator"]
    return CommonConditions(
        model_id=selected[0]["model_id"],
        temperature=selected[0]["temperature"],
        prompt_registry_version=next(iter(prompt_refs)),
        snapshot_id=emulator["data_snapshot_hash"],
        traps_enabled=emulator["traps_enabled"],
        latency_profile=emulator["latency_profile"],
        hr_employee_ids=tuple(str(value) for value in emulator["hr_employee_ids"]),
        code_execution=selected[0]["code_execution"],
    )


def select_modes(all_modes: Iterable[ModeConfig], names: str) -> dict[str, ModeConfig]:
    """Pick a named subset of arms, refusing a still-mock generated mode.

    Args:
        all_modes: Modes produced by ``build_modes``.
        names: Comma-separated mode names.

    Returns:
        Ordered mapping of requested name to config.

    Raises:
        ValueError: If a name is unknown, duplicated, or generated_skills is a mock.
    """
    requested = [item.strip() for item in names.split(",") if item.strip()]
    if not requested:
        raise ValueError("at least one benchmark mode is required")
    if len(requested) != len(set(requested)):
        raise ValueError("benchmark modes contain duplicates")
    available = {mode.name: mode for mode in all_modes}
    unknown = set(requested) - available.keys()
    if unknown:
        raise ValueError(f"unknown benchmark modes: {sorted(unknown)}")
    selected = {name: available[name] for name in requested}
    generated = selected.get(BenchmarkMode.GENERATED_SKILLS.value)
    if generated is not None and generated.is_mock:
        raise ValueError(
            "generated_skills is still a mock; provide and mount a generated catalog first"
        )
    return selected


def prepare(args: argparse.Namespace) -> PreparedBenchmark:
    """Validate the complete selected matrix without creating agent sessions.

    Args:
        args: Parsed CLI namespace (paths, modes, schema).

    Returns:
        Cases, live manifest, selected modes, activator and preflight results.

    Raises:
        ValueError: If cases, stand config or preflight disagree.
    """
    cases = load_cases_path(
        args.cases, schema_path=args.schema, limit=args.limit,
    )
    live = load_live_config(args.live_config)
    common = common_conditions(live)
    if any(case.raw["skill_registry_hash"] != live["skill_registry_hash"] for case in cases):
        mismatched = [
            case.case_id for case in cases
            if case.raw["skill_registry_hash"] != live["skill_registry_hash"]
        ]
        raise ValueError(
            "case skill_registry_hash differs from the live stand for: "
            + ", ".join(mismatched)
        )
    modes = build_modes(
        common,
        args.catalog,
        snapshots_root=args.catalog_snapshots,
    )
    selected = select_modes(modes, args.modes)
    refs = live["refs"]
    activator = PinnedConfigActivator(
        {name: refs[name] for name in selected},
        config_reader=lambda ref: live["configs"][ref],
    )

    # Validate every case x mode before the first model call.  Runner reuses
    # these immutable results instead of discovering a bad final case late.
    checked: dict[tuple[str, str], PreflightResult] = {}
    for mode in selected.values():
        activator.activate(mode)
        activator.deactivate(mode)
    for case in cases:
        for mode in selected.values():
            checked[(case.case_id, mode.name)] = preflight_case(
                case,
                mode,
                snapshot_root=args.data,
                standard_catalog_path=modes.existing_skills.catalog_path,
                model_catalog_path=args.model_catalog,
                agent_config_version=refs[mode.name],
                schema_path=args.schema,
            )

    return PreparedBenchmark(cases, live, selected, activator, checked)


def _eval_id(args: argparse.Namespace, default_prefix: str) -> str:
    prefix = args.eval_prefix or default_prefix
    return args.eval_id or (
        prefix + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )


def _manifest(
    args: argparse.Namespace,
    prepared: PreparedBenchmark,
    eval_id: str,
    *,
    check_only: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "eval_id": eval_id,
        "check_only": check_only,
        "cases_path": str(Path(args.cases).resolve()),
        "case_ids": [case.case_id for case in prepared.cases],
        "modes": [asdict(mode) for mode in prepared.selected_modes.values()],
        "repetitions": 0 if check_only else args.repetitions,
        "live_stand": prepared.live,
    }


def check(args: argparse.Namespace) -> ResultWriter:
    """Run all preflight checks and persist a report; never call an agent.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Writer whose directory contains ``run-manifest.json`` and ``preflight.json``.
    """
    prepared = prepare(args)
    eval_id = _eval_id(args, "check")
    writer = ResultWriter(args.results, eval_id)
    writer.write_manifest(_manifest(args, prepared, eval_id, check_only=True))
    report = [
        {
            "case_id": result.case_id,
            "mode": result.mode,
            "status": result.status,
            "fingerprint": (
                result.fingerprint.as_dict() if result.fingerprint else None
            ),
            "condition_id": (
                result.fingerprint.condition_id if result.fingerprint else None
            ),
        }
        for _key, result in sorted(prepared.checked.items())
    ]
    (writer.root / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return writer


def run(
    args: argparse.Namespace, *, prepared: PreparedBenchmark | None = None,
) -> tuple[list[Any], ResultWriter]:
    """Execute the prepared matrix against a local or remote stand.

    Args:
        args: Parsed CLI namespace including timeout and repetitions.
        prepared: Optional remote preflight; otherwise validate local snapshots.

    Returns:
        Pair of run results and the writer that stored them.
    """
    prepared = prepared or prepare(args)
    eval_id = _eval_id(args, "benchmark")
    writer = ResultWriter(args.results, eval_id)
    writer.write_manifest(_manifest(args, prepared, eval_id, check_only=False))
    runner = BenchmarkRunner(
        eval_id,
        preflight=lambda case, mode: prepared.checked[(case.case_id, mode.name)],
        activator=prepared.activator,
        executor=StandSessionExecutor(
            env_file=args.env_file,
            timeout=args.timeout,
            trace_attempts=args.trace_attempts,
            trace_backend=getattr(args, "trace_backend", "research"),
            **({"public_url": args.public_url} if getattr(args, "public_url", None) else {}),
        ),
        writer=writer,
    )
    return runner.run(
        prepared.cases, prepared.selected_modes, repetitions=args.repetitions,
    ), writer


def parser() -> argparse.ArgumentParser:
    """Build the host-side benchmark CLI.

    Returns:
        Parser for ``python -m sim.benchmark.cli``.
    """
    result = argparse.ArgumentParser(
        description="Run ready benchmark cases against the local Compose stand."
    )
    result.add_argument("--cases", required=True, help="Directory, JSON case, or JSONL suite")
    result.add_argument("--data", default="data-small", help="Host path to the mounted data snapshot")
    result.add_argument("--catalog", default="heimdall-skills")
    result.add_argument("--catalog-snapshots", default="var/benchmark-catalog-snapshots")
    result.add_argument("--model-catalog", default="catalog/snapshot.json")
    result.add_argument("--schema", default=SCHEMA_RELATIVE)
    result.add_argument("--live-config", default="var/benchmark-live-config.json")
    result.add_argument("--results", default="benchmarking/results")
    result.add_argument("--modes", default=",".join(DEFAULT_MODES))
    result.add_argument("--limit", type=int, help="Run/check only the first N ready cases")
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--eval-id")
    result.add_argument("--eval-prefix")
    result.add_argument("--check-only", action="store_true")
    result.add_argument("--env-file", default=ENV_RELATIVE)
    result.add_argument("--timeout", type=float, default=1800.0)
    result.add_argument(
        "--trace-attempts", type=int, default=30,
        help="Give up after this many unchanged incomplete Phoenix snapshots; "
        "a growing trace keeps polling",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Entry point for check-only or live smoke runs.

    Args:
        argv: Optional argument vector; ``None`` uses ``sys.argv``.

    Returns:
        ``0`` after a finished matrix. Unscored or invalid-condition cells are
        listed in ``summary.json`` and do not fail the process. ``2`` if
        preflight or setup failed before any LLM call.
    """
    try:
        args = parser().parse_args(argv)
        if args.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if args.check_only:
            writer = check(args)
            print(json.dumps({
                "results_dir": str(writer.root),
                "preflight": str(writer.root / "preflight.json"),
                "agent_calls": 0,
            }, ensure_ascii=False, indent=2))
            return 0
        results, writer = run(args)
        print(json.dumps(_finished_report(results, writer), ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"benchmark failed before completion: {exc}", file=sys.stderr)
        return 2


def _finished_report(results: list[Any], writer: ResultWriter) -> dict[str, Any]:
    """Counts for a matrix that ran to the end; review cells are not failures.

    Args:
        results: All matrix cells, including skips and review statuses.
        writer: Destination that already contains ``summary.json``.

    Returns:
        JSON-serializable report printed to stdout.
    """
    summary = summarize_results(results)
    return {
        "results_dir": str(writer.root),
        "runs": len(results),
        "n_completed": summary["n_runs"],
        "n_review": len(summary["review"]),
        "n_normalization_pending": summary["n_normalization_pending"],
        "n_condition_invalid": summary["n_condition_invalid"],
        "n_unscored": summary["n_unscored"],
        "summary": str(writer.root / "summary.json"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
