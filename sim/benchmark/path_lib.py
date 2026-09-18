"""Default filesystem locations for the skill benchmark.

``PROJECT_ROOT`` is the repository root. Every other path is relative to it.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_RELATIVE = "benchmarking/schemas/benchmark-case-v3.schema.json"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / SCHEMA_RELATIVE

RESPONSE_PROMPT_RELATIVE = "sim/benchmark/prompts/benchmark-response-prompt.txt"
RESPONSE_PROMPT_PATH = PROJECT_ROOT / RESPONSE_PROMPT_RELATIVE
