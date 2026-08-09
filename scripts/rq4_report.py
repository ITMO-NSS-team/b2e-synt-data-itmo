#!/usr/bin/env python
"""CLI entry point for the RQ4 report — thin wrapper over
``sim.research.report``, which holds the actual logic (health, guard,
primary, learning curve, cost, limits) and its own test suite. Kept as a
separate script, not a ``__main__`` block inside the package module, to
match this repo's existing convention (``scripts/run_rq4.py`` is the driver;
``scripts/rq4_report.py`` is the reader) and so ``python scripts/rq4_report.py
--run-id ...`` is the one command a sleepy researcher needs to remember.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.research.report import main  # noqa: E402

if __name__ == "__main__":
    main()
