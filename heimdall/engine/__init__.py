"""Исполнитель тела ``mcp_query`` поверх колоночного снимка."""
from .errors import ERRORS, HeimdallError, fail
from .quirks import ALL_QUIRKS, Quirks

__all__ = ["ERRORS", "HeimdallError", "fail", "Quirks", "ALL_QUIRKS"]
