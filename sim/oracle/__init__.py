"""Reference answers for the question basket.

The oracle is deliberately a separate package from the emulator. The emulator
serves the corpus; the oracle knows the answers. Nothing here may ever be
importable from the request path — a gold label that leaks into an API response
stops being a measurement and becomes the thing being measured.
"""
from __future__ import annotations

from .labels import (GoldLabels, VacancySpec, as_dict, IMPACT_WEIGHTS,
                     KEY_IMPACT_PCT, READINESS_THRESHOLDS)

__all__ = ["GoldLabels", "VacancySpec", "as_dict", "IMPACT_WEIGHTS",
           "KEY_IMPACT_PCT", "READINESS_THRESHOLDS"]
