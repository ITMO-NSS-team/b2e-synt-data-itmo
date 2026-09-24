"""Shared constants for added benchmark tests."""
from __future__ import annotations

from tests.path_lib import EXAMPLE, MODEL_CATALOG, SCHEMA

EMPTY_HASH = "sha256:" + "0" * 64

__all__ = ["EMPTY_HASH", "EXAMPLE", "MODEL_CATALOG", "SCHEMA"]
