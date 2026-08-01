"""Simulation environment for controlled B2E agent experiments.

The corpus in ``b2e/`` and the API emulator in ``heimdall/`` are the measuring
instrument. This package is the laboratory built around it: services that let a
researcher vary one thing, run a question, and compare the traces.

Layout
------
``sim.fingerprint``  run identity — the thing that makes two runs comparable
``sim.latency``      latency profiles for the emulator (RQ2 depends on these)
``sim.registry``     versioned, content-addressed config store, append-only
``sim.emulator``     C1, Heimdall API over the local snapshot
``sim.agent``        C2, the agent under test
``sim.research``     C4, read/write API for researchers
``sim.admin``        C5, operator UI and the skill approval gate
``sim.sandbox``      skill execution, isolated (see docs/skill-execution-threat-model.md)
``sim.oracle``       gold labels and the question basket
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
