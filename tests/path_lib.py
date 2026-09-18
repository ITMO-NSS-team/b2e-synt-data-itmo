"""Repository root for tests. Other paths are relative to this directory."""
from __future__ import annotations

from sim.benchmark.path_lib import DEFAULT_SCHEMA_PATH, PROJECT_ROOT

EXAMPLE = PROJECT_ROOT / "tests/fixtures/benchmark-case-v3.json"
SCHEMA = DEFAULT_SCHEMA_PATH
MODEL_CATALOG = PROJECT_ROOT / "catalog/snapshot.json"
