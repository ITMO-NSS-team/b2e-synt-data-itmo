"""Repository root for tests. Other paths are relative to this directory."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXAMPLE = PROJECT_ROOT / "tests/fixtures/benchmark-case-v3.json"
SCHEMA = PROJECT_ROOT / "benchmarking/schemas/benchmark-case-v3.schema.json"
MODEL_CATALOG = PROJECT_ROOT / "catalog/snapshot.json"
