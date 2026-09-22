"""Start the benchmark beside a remote stand and download its artifacts.

The server checkout owns execution. This local command only invokes the
server-side Make target over SSH and copies the completed result directory.
It never uploads code, cases or configuration and never changes Git state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile

from .remote import close_ssh_master, ssh_command

_SAFE_EVAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _relative_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ValueError(f"{label} must be a path inside the server checkout")
    return path.as_posix()


def _remote_root(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("--remote-root must be an absolute normalized path")
    return path.as_posix()


def server_make_command(args: argparse.Namespace) -> tuple[str, str]:
    """Build a quoted server command and return its deterministic eval id."""
    cases = _relative_path(args.cases, label="--cases")
    remote_results = _relative_path(args.remote_results, label="--remote-results")
    root = _remote_root(args.remote_root)
    eval_id = args.eval_id or (
        "server-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if not _SAFE_EVAL_ID.fullmatch(eval_id):
        raise ValueError("--eval-id contains unsupported characters")
    make = [
        "make",
        "benchmark-server",
        f"CASES={cases}",
        f"BENCH_RESULTS={remote_results}",
        f"BENCH_MODES={args.modes}",
        f"BENCH_REPETITIONS={args.repetitions}",
        f"BENCH_TIMEOUT={args.timeout}",
        f"BENCH_TRACE_TIMEOUT={args.trace_timeout}",
        f"BENCH_EVAL_ID={eval_id}",
    ]
    if args.limit is not None:
        make.append(f"BENCH_LIMIT={args.limit}")
    if args.model:
        make.append(f"BENCH_MODEL={args.model}")
    return f"cd {shlex.quote(root)} && {shlex.join(make)}", eval_id


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run benchmark in the remote stand network and download results.",
    )
    result.add_argument("--cases", required=True)
    result.add_argument("--ssh", default="nnikitin@10.32.1.71")
    result.add_argument(
        "--remote-root", default="/var/essdata/b2e-synt-data-itmo",
    )
    result.add_argument("--remote-results", default="benchmarking/results")
    result.add_argument("--local-results", default="benchmarking/results")
    result.add_argument(
        "--modes", default="general_knowledge,existing_skills",
    )
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--limit", type=int)
    result.add_argument("--timeout", type=float, default=1800.0)
    result.add_argument("--trace-timeout", type=float, default=300.0)
    result.add_argument("--model")
    result.add_argument("--eval-id")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if not args.ssh or args.ssh.startswith("-"):
            raise ValueError("--ssh must be a hostname or user@hostname")
        if args.repetitions < 1:
            raise ValueError("--repetitions must be >= 1")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be >= 1")
        if args.timeout <= 0 or args.trace_timeout <= 0:
            raise ValueError("timeouts must be positive")
        command, eval_id = server_make_command(args)
        remote_results = _relative_path(
            args.remote_results, label="--remote-results",
        )
        remote_root = _remote_root(args.remote_root)
        local_root = Path(args.local_results).resolve()
        destination = local_root / eval_id
        if destination.exists():
            raise ValueError(f"local result directory already exists: {destination}")
        local_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="b2e-server-bench-", dir="/tmp") as temp:
            control_path = str(Path(temp) / "ssh")
            try:
                print(
                    "Running benchmark inside the remote stand network. "
                    "The server checkout must contain this benchmark version.",
                    flush=True,
                )
                subprocess.run(
                    ssh_command(args.ssh, control_path, command),
                    check=True,
                )
                source_path = PurePosixPath(remote_root) / remote_results / eval_id
                subprocess.run(
                    [
                        "scp", "-r",
                        "-o", "ControlMaster=auto",
                        "-o", f"ControlPath={control_path}",
                        f"{args.ssh}:{source_path.as_posix()}",
                        str(local_root),
                    ],
                    check=True,
                )
            finally:
                close_ssh_master(args.ssh, control_path)
        print(json.dumps({
            "eval_id": eval_id,
            "results_dir": str(destination),
            "execution": "remote_server",
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"remote server benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
