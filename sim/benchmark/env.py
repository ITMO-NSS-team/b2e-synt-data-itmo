"""Load local ``.env`` into the process environment. No code defaults."""
from __future__ import annotations

import os
from pathlib import Path

from .path_lib import ENV_PATH


def load_env(path: str | Path | None = None) -> None:
    """Fill missing ``os.environ`` keys from a local ``.env``.

    Already-set process variables (Compose, Make) are left alone. A missing
    file is allowed.

    Args:
        path: Env file. ``None`` uses ``deploy/.env`` at the repository root.
    """
    env_path = Path(path) if path is not None else ENV_PATH
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        key = name.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'").strip('"')


def require_env(key: str) -> str:
    """Return ``os.getenv(key)`` or raise if it is missing or blank.

    Args:
        key: Variable name.

    Returns:
        Stripped value.

    Raises:
        ValueError: If ``key`` is unset or blank.
    """
    value = (os.getenv(key) or "").strip()
    if not value:
        raise ValueError(
            f"{key} is empty. Set it in {ENV_PATH} or the process environment."
        )
    return value
