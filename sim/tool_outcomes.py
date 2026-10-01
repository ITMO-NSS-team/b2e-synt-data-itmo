"""CLI tool-result classification shared by tracing and benchmark readers."""
from __future__ import annotations

from typing import Any


def permission_denied(output: Any) -> bool:
    """Recognize native CLI refusals, not data-plane HTTP 403/access errors."""
    if isinstance(output, dict):
        # Keep this specific to native permission errors, not a generic
        # backend's `permission_denied` code or arbitrary prose in its data.
        return output.get("name") in ("PermissionDeniedError", "PermissionRejectedError")
    if not isinstance(output, str):
        if isinstance(output, list):
            return any(permission_denied(block.get("text")) for block in output
                       if isinstance(block, dict) and block.get("type") == "text")
        return False
    text = output.strip().lower()
    return (
        text.startswith("the user has specified a rule which prevents you from using this specific tool call")
        or (text.startswith("permission to use ") and "has been denied" in text)
    )


def tool_outcome(output: Any, *, is_error: bool = False,
                 denied: bool = False, unfinished: bool = False) -> dict[str, Any]:
    """Preserve structured failure evidence, with wording fallback for old CLIs."""
    denied = denied or permission_denied(output)
    failed = bool(is_error or denied or unfinished)
    return {"is_error": failed, "permission_denied": denied,
            "status": ("unfinished" if unfinished else "permission_denied" if denied
                       else "error" if failed else "completed")}
