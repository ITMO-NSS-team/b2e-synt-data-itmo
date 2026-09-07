from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SKILL_FACTORY = _ROOT / "skill-factory"
if _SKILL_FACTORY.is_dir() and str(_SKILL_FACTORY) not in sys.path:
    sys.path.insert(0, str(_SKILL_FACTORY))
