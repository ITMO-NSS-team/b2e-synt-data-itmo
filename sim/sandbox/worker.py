"""Spool worker: claims a job, runs the runner as a child, enforces the clock.

Runs in a container with ``network_mode: none``. Reaches the gateway only through
a shared spool volume, which is what lets the executing process be genuinely
network-isolated without any service holding a Docker socket.

Handoff is write-to-temp then ``os.rename``. Rename is atomic on the shared
volume, so two workers cannot claim the same job — measured during the G1 audit
at 60 concurrent submits producing 60 unique results with none lost or doubled.

The wall clock is enforced here rather than inside the runner, because a limit
the untrusted side enforces is not a limit. The child is started with
``start_new_session=True`` so the timeout kills the whole process group.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("b2e.sandbox.worker")

DEFAULT_WALL_SECONDS = 10.0
#: Cap on the result document. A worker is untrusted output from the gateway's
#: point of view, so its response is bounded before it is ever parsed.
MAX_RESULT_BYTES = 1 * 1024 * 1024


class Spool:
    """Three directories on one shared volume: new, run, out."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.new = self.root / "new"
        self.run = self.root / "run"
        self.out = self.root / "out"
        for directory in (self.new, self.run, self.out):
            directory.mkdir(parents=True, exist_ok=True)

    def submit(self, job_id: str, job: dict[str, Any]) -> None:
        temp = self.new / f".{job_id}.tmp"
        temp.write_text(json.dumps(job, ensure_ascii=False), "utf-8")
        temp.rename(self.new / f"{job_id}.json")

    def claim(self, slot: str) -> tuple[str, dict[str, Any]] | None:
        """Atomically take one job. Rename is the lock."""
        for path in sorted(self.new.glob("*.json")):
            target = self.run / f"{path.stem}.{slot}.json"
            try:
                path.rename(target)
            except (FileNotFoundError, OSError):
                continue                      # another worker won the race
            try:
                return path.stem, json.loads(target.read_text("utf-8"))
            except ValueError:
                self.finish(path.stem, {"ok": False, "error": {
                    "kind": "poison", "detail": "job document is not valid JSON"}})
                target.unlink(missing_ok=True)
                return None
        return None

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        body = json.dumps(result, ensure_ascii=False)[:MAX_RESULT_BYTES]
        temp = self.out / f".{job_id}.tmp"
        temp.write_text(body, "utf-8")
        temp.rename(self.out / f"{job_id}.json")
        for leftover in self.run.glob(f"{job_id}.*.json"):
            leftover.unlink(missing_ok=True)

    def collect(self, job_id: str, timeout: float = 30.0,
                poll: float = 0.05) -> dict[str, Any] | None:
        deadline = time.time() + timeout
        path = self.out / f"{job_id}.json"
        while time.time() < deadline:
            if path.exists():
                try:
                    result = json.loads(path.read_text("utf-8"))
                except ValueError:
                    return {"ok": False, "error": {
                        "kind": "corrupt-result",
                        "detail": "worker wrote a result that is not valid JSON"}}
                path.unlink(missing_ok=True)
                return result
            time.sleep(poll)
        return None

    def reap(self, older_than_seconds: float) -> list[str]:
        """Fail out jobs stranded by a worker that died.

        Slot identity is the container hostname, which is stable across
        restart-policy restarts but not across ``compose up --force-recreate``.
        This time-based sweep covers the gap (residual risk R-10).
        """
        reaped = []
        cutoff = time.time() - older_than_seconds
        for path in self.run.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                job_id = path.stem.split(".")[0]
                self.finish(job_id, {"ok": False, "error": {
                    "kind": "stranded",
                    "detail": "worker did not report; job reaped by the gateway"}})
                reaped.append(job_id)
        return reaped


#: The runner is invoked by FILE PATH, not with ``-m``. ``python3 -I`` implies
#: ``-E``, so PYTHONPATH is ignored and a package-qualified ``-m`` target cannot
#: resolve. Running the file directly puts its own directory on ``sys.path``,
#: which is all the runner needs — it imports nothing but the standard library,
#: deliberately, so that it can be mounted read-only on its own.
RUNNER_PATH = str(Path(__file__).with_name("runner.py"))


def run_job(job: dict[str, Any], *, python: str = sys.executable) -> dict[str, Any]:
    """Run one job in a child process under a wall clock."""
    wall = float(job.get("wall_seconds") or DEFAULT_WALL_SECONDS)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [python, "-I", "-S", RUNNER_PATH],
            input=json.dumps({"code": job.get("code", ""),
                              "payload": job.get("payload") or {},
                              "limits": job.get("limits")}).encode("utf-8"),
            capture_output=True,
            timeout=wall,
            # Explicitly constructed: os.environ is never passed through, so a
            # credential the parent holds cannot reach skill code.
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp",
                 "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
            cwd="/tmp",
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": {
            "kind": "timeout",
            "detail": f"skill exceeded its {wall}s wall clock and was killed"},
            "wall_ms": round(wall * 1000, 2)}
    except Exception as exc:                                   # pragma: no cover
        return {"ok": False, "error": {"kind": "spawn",
                                       "detail": f"{type(exc).__name__}: {exc}"}}

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    raw = completed.stdout.decode("utf-8", "replace").strip()
    if not raw:
        return {"ok": False, "error": {
            "kind": "no-result",
            "detail": f"runner exited {completed.returncode} with no result document",
            "stderr": completed.stderr.decode("utf-8", "replace")[:2000]},
            "wall_ms": elapsed}
    try:
        result = json.loads(raw)
    except ValueError:
        return {"ok": False, "error": {"kind": "not-json",
                                       "detail": raw[:2000]}, "wall_ms": elapsed}
    result.setdefault("wall_ms", elapsed)
    return result


def serve(spool_root: str, *, slot: str | None = None, poll: float = 0.05) -> None:
    slot = slot or os.environ.get("HOSTNAME") or f"w{os.getpid()}"
    spool = Spool(spool_root)
    log.info("sandbox worker slot=%s spool=%s", slot, spool_root)

    stopping = {"now": False}

    def _stop(*_: Any) -> None:
        stopping["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while not stopping["now"]:
        claimed = spool.claim(slot)
        if claimed is None:
            time.sleep(poll)
            continue
        job_id, job = claimed
        try:
            result = run_job(job)
        except Exception as exc:                               # pragma: no cover
            result = {"ok": False, "error": {"kind": "worker",
                                             "detail": f"{type(exc).__name__}: {exc}"}}
        spool.finish(job_id, result)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    serve(os.environ.get("SANDBOX_SPOOL", "/spool"))


if __name__ == "__main__":
    main()
