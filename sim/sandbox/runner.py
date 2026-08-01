"""The innermost process: runs exactly one skill and exits.

Executed as ``python3 -I -S -m sim.sandbox.runner`` with a hand-built
environment, inside a container that has an empty network namespace, a read-only
root filesystem and dropped capabilities. It reads one JSON job on stdin and
writes one JSON result on stdout.

What is a boundary here and what is not
---------------------------------------
**Boundaries** (kernel-enforced, outside this file): the empty netns, the
read-only rootfs, the cgroup memory/CPU/pids caps, the dropped capabilities, the
scrubbed environment, and the separate PID namespace.

**Not boundaries** (in this file): the import allowlist and the rlimits. They are
here for telemetry, for catching honest mistakes in agent-authored skills, and
for forcing an author to declare a capability surface a reviewer can read. A
hostile skill removes the allowlist with ``sys.meta_path[:] = []``, and this was
demonstrated during the G1 audit, along with ``ctypes`` reaching a raw
``syscall(41)``. Nothing in this file should ever be described as containment.

stdout discipline
-----------------
The first thing ``main`` does is move the real stdout aside and point fd 1 at
stderr, so a ``print()`` inside skill code lands in the logs instead of
corrupting the result document. The result is written to the saved descriptor at
exit. Without this a single stray print turns a valid result into a parse error.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from typing import Any

#: Modules a skill may import. Widening this list is a reviewable act: each
#: addition is a small, permanent erosion (residual risk R-9).
ALLOWED_IMPORTS = frozenset({
    "json", "math", "statistics", "datetime", "decimal", "fractions", "re",
    "collections", "itertools", "functools", "operator", "typing",
    "dataclasses", "enum", "uuid", "hashlib", "textwrap", "unicodedata",
    "abc", "numbers", "copy", "string", "bisect", "heapq", "array",
})

DEFAULT_LIMITS = {
    "as_bytes": 192 * 1024 * 1024,   # soft == hard; a clean MemoryError
    "cpu_seconds": 5,                # soft == hard; SIGXCPU is catchable
    "fsize_bytes": 16 * 1024 * 1024,
    "nofile": 64,
    "nproc": 0,                      # no fork, no subprocess
}


class ImportBlocked(ImportError):
    pass


class _AllowlistFinder:
    """A meta_path finder that refuses anything outside the allowlist.

    Records what it refused so the rejection shows up in the trace as
    ``b2e.sandbox.rejected_imports`` — which is its actual value.
    """

    def __init__(self) -> None:
        self.rejected: list[str] = []

    def find_module(self, fullname: str, path: Any = None) -> None:
        return None

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        root = fullname.split(".")[0]
        if root in ALLOWED_IMPORTS or root.startswith("_"):
            return None
        self.rejected.append(fullname)
        raise ImportBlocked(
            f"import of {fullname!r} is not in the skill allowlist. "
            f"Declare only what you need; the sandbox has no network and no "
            f"filesystem beyond a scratch directory."
        )


def apply_rlimits(limits: dict[str, int]) -> None:
    """Set every limit with soft == hard.

    Soft-only limits are useless against hostile code: an unprivileged process
    may raise its own soft limit to the hard limit at will, and ``SIGXCPU`` is
    catchable. Setting both is the difference between a speed bump and a stop.
    """
    pairs = [
        (resource.RLIMIT_AS, limits["as_bytes"]),
        (resource.RLIMIT_CPU, limits["cpu_seconds"]),
        (resource.RLIMIT_FSIZE, limits["fsize_bytes"]),
        (resource.RLIMIT_NOFILE, limits["nofile"]),
        (resource.RLIMIT_CORE, 0),
    ]
    if hasattr(resource, "RLIMIT_NPROC"):
        pairs.append((resource.RLIMIT_NPROC, limits["nproc"]))
    for which, value in pairs:
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            # A limit we cannot set is reported, never silently skipped: the
            # caller decides whether to proceed.
            print(f"[sandbox] could not set rlimit {which}", file=sys.stderr)


def execute(code: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Compile and run a skill's ``run(payload)`` in a stripped namespace.

    Deliberately does **not** apply rlimits. They are process-wide and
    irreversible, so applying them here would poison whatever process imported
    this module — which is exactly what happened the first time this was written,
    when ``RLIMIT_NPROC=0`` inside ``execute`` left the *test runner* unable to
    fork. Limits belong in :func:`main`, which runs in the disposable child and
    nowhere else.

    Note also what the allowlist cannot do: a module already present in
    ``sys.modules`` is returned without ``meta_path`` ever being consulted. In
    the real runner (``python3 -I -S``) almost nothing is preloaded, but this is
    a further reason the allowlist is telemetry rather than containment.
    """
    finder = _AllowlistFinder()
    sys.meta_path.insert(0, finder)

    started = time.perf_counter()
    namespace: dict[str, Any] = {"__name__": "skill", "__builtins__": __builtins__}
    try:
        compiled = compile(code, "<skill>", "exec")
        exec(compiled, namespace)                              # noqa: S102
        entry = namespace.get("run")
        if not callable(entry):
            return _error("no-entrypoint", "skill defines no callable run(payload)",
                          finder, started)
        result = entry(payload)
        # Strict serialisation. `default=str` would silently coerce a live object
        # to "<... object ...>" and call it a result (residual risk R-6).
        json.dumps(result)
        return {"ok": True, "result": result,
                "wall_ms": round((time.perf_counter() - started) * 1000, 2),
                "rejected_imports": finder.rejected,
                "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    except ImportBlocked as exc:
        return _error("import-blocked", str(exc), finder, started)
    except MemoryError:
        return _error("memory", "skill exceeded its memory limit", finder, started)
    except TypeError as exc:
        return _error("not-json", f"result is not JSON-serialisable: {exc}",
                      finder, started)
    except Exception as exc:
        return _error(type(exc).__name__, str(exc)[:2000], finder, started)
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


def _error(kind: str, detail: str, finder: _AllowlistFinder,
           started: float) -> dict[str, Any]:
    return {"ok": False, "error": {"kind": kind, "detail": detail},
            "wall_ms": round((time.perf_counter() - started) * 1000, 2),
            "rejected_imports": finder.rejected}


def main() -> int:
    # Move the real stdout aside before any skill code can print to it.
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        job = json.loads(sys.stdin.read() or "{}")
        # Limits are applied here, in the child that is about to die, and never
        # in a process that has to keep working afterwards.
        apply_rlimits({**DEFAULT_LIMITS, **(job.get("limits") or {})})
        result = execute(job.get("code", ""), job.get("payload") or {})
    except Exception as exc:                                   # pragma: no cover
        result = {"ok": False,
                  "error": {"kind": "runner", "detail": f"{type(exc).__name__}: {exc}"}}
    os.write(saved, json.dumps(result, ensure_ascii=False).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
