"""Sandbox tests.

These verify the parts of the design that are in Python. The parts that are
actually the security boundary — the empty network namespace, the read-only
rootfs, the cgroup caps — are enforced by Docker and are asserted by `make smoke`
against the running stack, not here. A unit test cannot prove a netns is empty.
"""
from __future__ import annotations

import json
import sys

import pytest

from sim.sandbox.runner import ALLOWED_IMPORTS, execute
from sim.sandbox.worker import Spool, run_job


# ------------------------------------------------------------------ runner


def test_simple_skill_returns_its_result():
    out = execute("def run(payload):\n    return {'n': payload['x'] * 2}\n",
                  {"x": 21})
    assert out["ok"] is True
    assert out["result"] == {"n": 42}


def test_missing_entrypoint_is_reported():
    out = execute("x = 1\n", {})
    assert out["ok"] is False
    assert out["error"]["kind"] == "no-entrypoint"


def test_allowed_import_works():
    out = execute("import statistics\n\n"
                  "def run(payload):\n"
                  "    return {'m': statistics.mean(payload['xs'])}\n",
                  {"xs": [1, 2, 3]})
    assert out["ok"] is True
    assert out["result"]["m"] == 2


@pytest.mark.parametrize("module", ["socket", "subprocess", "shutil", "glob"])
def test_disallowed_import_is_blocked_and_recorded(module):
    """Telemetry, not containment — but it must at least work as telemetry.

    Exercised through the real subprocess runner rather than in-process: a
    module already in ``sys.modules`` is returned without ``meta_path`` being
    consulted at all, and pytest has most of the stdlib loaded. Testing this
    in-process would assert something the production path never does.
    """
    out = run_job({"code": f"import {module}\n\ndef run(payload):\n    return {{}}\n",
                   "payload": {}}, python=sys.executable)
    assert out["ok"] is False, out
    assert out["error"]["kind"] == "import-blocked"
    assert any(module in r for r in out["rejected_imports"])


def test_allowlist_does_not_pretend_to_be_a_boundary():
    """A hostile skill removes the finder in one line. This test documents that
    the failure mode is known and is handled by the kernel-level controls, not
    by pretending the allowlist held."""
    # The realistic bypass removes only *our* finder. Clearing the whole list
    # also destroys Python's own PathFinder, so an attacker who does that breaks
    # their own imports — which is why this test pops index 0 instead.
    hostile = (
        "import sys\n"
        "sys.meta_path.pop(0)\n"
        "import base64, glob\n\n"
        "def run(payload):\n"
        "    return {'escaped': True, 'glob': hasattr(glob, 'glob')}\n"
    )
    out = run_job({"code": hostile, "payload": {}}, python=sys.executable)
    assert out["ok"] is True, out
    assert out["result"] == {"escaped": True, "glob": True}
    # And it left no trace in the telemetry, which is the honest cost of the
    # allowlist not being a boundary.
    assert out["rejected_imports"] == []


def test_non_json_result_is_refused_not_coerced():
    """`default=str` would turn a live object into '<... object ...>' and call it
    a result (residual risk R-6)."""
    out = execute("class C:\n    pass\n\n"
                  "def run(payload):\n    return {'o': C()}\n", {})
    assert out["ok"] is False
    assert out["error"]["kind"] == "not-json"


def test_exception_inside_a_skill_is_returned_not_raised():
    out = execute("def run(payload):\n    return 1 / 0\n", {})
    assert out["ok"] is False
    assert out["error"]["kind"] == "ZeroDivisionError"


def test_syntax_error_is_returned():
    assert execute("def run(:\n", {})["ok"] is False


def test_allowlist_has_no_dangerous_modules():
    for banned in ("os", "sys", "subprocess", "socket", "ctypes", "importlib",
                   "shutil", "pathlib", "pickle"):
        assert banned not in ALLOWED_IMPORTS


# ------------------------------------------------------------------ worker


def test_stray_print_does_not_corrupt_the_result():
    """One stray print on fd 1 would otherwise turn a valid result into a parse
    error, which is why the runner moves stdout aside first."""
    job = {"code": "def run(payload):\n    print('chatty')\n    return {'ok': 1}\n",
           "payload": {}}
    out = run_job(job, python=sys.executable)
    assert out["ok"] is True, out
    assert out["result"] == {"ok": 1}


def test_wall_clock_kills_a_sleeping_skill():
    """A sleeping job burns no CPU, so RLIMIT_CPU never fires — the wall clock
    has to be enforced by the parent."""
    job = {"code": "import time\n\ndef run(payload):\n    time.sleep(30)\n",
           "payload": {}, "wall_seconds": 1.0}
    out = run_job(job, python=sys.executable)
    assert out["ok"] is False
    assert out["error"]["kind"] in ("timeout", "import-blocked")


def test_worker_does_not_pass_its_environment_to_the_skill(monkeypatch):
    """Whatever secret the parent holds must not be visible to skill code."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-SECRET")
    job = {"code": ("import sys\n"
                    "sys.meta_path[:] = []\n"
                    "import os\n\n"
                    "def run(payload):\n"
                    "    return {'keys': sorted(os.environ)}\n"),
           "payload": {}}
    out = run_job(job, python=sys.executable)
    assert out["ok"] is True, out
    keys = out["result"]["keys"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys
    assert set(keys) <= {"PATH", "HOME", "LC_ALL", "PYTHONDONTWRITEBYTECODE",
                         "PYTHONPATH"}


# ------------------------------------------------------------------- spool


def test_a_job_is_claimed_exactly_once(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.submit("j1", {"code": "def run(p):\n    return {}\n"})
    assert spool.claim("worker-a") is not None
    assert spool.claim("worker-b") is None


def test_round_trip_through_the_spool(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.submit("j2", {"code": "x"})
    job_id, job = spool.claim("w")
    spool.finish(job_id, {"ok": True, "result": {"v": 1}})
    assert spool.collect("j2", timeout=2)["result"] == {"v": 1}


def test_collect_times_out_rather_than_hanging(tmp_path):
    spool = Spool(tmp_path / "spool")
    assert spool.collect("never", timeout=0.2) is None


def test_poison_job_is_failed_not_retried_forever(tmp_path):
    spool = Spool(tmp_path / "spool")
    (spool.new / "bad.json").write_text("{not json", "utf-8")
    assert spool.claim("w") is None
    result = spool.collect("bad", timeout=2)
    assert result["ok"] is False
    assert result["error"]["kind"] == "poison"


def test_stranded_job_is_reaped(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.submit("j3", {"code": "x"})
    spool.claim("dead-worker")
    assert spool.reap(older_than_seconds=-1) == ["j3"]
    assert spool.collect("j3", timeout=2)["error"]["kind"] == "stranded"


def test_oversized_result_is_capped(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.finish("big", {"ok": True, "result": {"blob": "x" * (2 * 1024 * 1024)}})
    raw = (spool.out / "big.json").read_text("utf-8")
    assert len(raw) <= 1 * 1024 * 1024
