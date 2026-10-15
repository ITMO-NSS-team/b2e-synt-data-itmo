"""Provision and retain private session catalogs without persisting capabilities."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from math import isfinite
import threading
import time

from fastapi import HTTPException
import httpx


@dataclass
class CatalogBinding:
    capability: str
    expires_at: float


class SessionCatalogs:
    def __init__(self, base_url: str, admin_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_key = admin_key
        self._contexts: dict[str, CatalogBinding] = {}
        self.ttl_seconds = float(os.getenv("HEIMDALL_CATALOG_TTL_SECONDS", "86400"))
        if not isfinite(self.ttl_seconds) or self.ttl_seconds <= 0:
            raise ValueError("catalog TTL must be positive")
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._closing: set[str] = set()

    def _request(self, method: str, *, context: str | None = None, **kwargs):
        if not self.admin_key:
            raise HTTPException(503, "HEIMDALL_CATALOG_ADMIN_KEY is required for extra skills")
        headers = {"X-Catalog-Admin-Key": self.admin_key}
        if context is not None:
            headers["X-Skill-Catalog-Context"] = context
        path = "/internal/skill-catalogs" + ("/current" if context is not None else "")
        try:
            response = httpx.request(method, self.base_url + path, headers=headers,
                                     timeout=30, trust_env=False, **kwargs)
        except httpx.HTTPError as exc:
            raise HTTPException(503, "isolated catalog service unavailable") from exc
        if not response.is_success:
            # Never put capability-bearing request headers into error receipts.
            try:
                detail = response.json().get("detail", "isolated catalog request failed")
            except (ValueError, AttributeError):
                detail = "isolated catalog request failed"
            raise HTTPException(response.status_code, detail)
        return response.json() if response.content else None

    def provision(self, employee_id: str, extra_skills: list[dict]) -> dict:
        return self._request("POST", json={"employee_id": employee_id,
                                          "extra_skills": extra_skills})

    def _reap(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, binding in self._contexts.items()
                   if binding.expires_at <= now
                   and sid not in self._active and sid not in self._closing]
        for sid in expired:
            del self._contexts[sid]

    def bind(self, session_id: str, context: str) -> None:
        with self._lock:
            self._reap()
            self._contexts[session_id] = CatalogBinding(
                context, time.monotonic() + self.ttl_seconds)

    def context_for(self, session: dict) -> str | None:
        if "skill_catalog" not in session["fingerprint"]:
            return None
        with self._lock:
            binding = self._contexts.get(session["id"])
        if binding is None:
            raise HTTPException(410, "isolated session interrupted or closed; start a new session")
        try:
            self._request("GET", context=binding.capability,
                          params={"employee_id": session["employee_id"]})
        except HTTPException as exc:
            if exc.status_code == 410:
                with self._lock:
                    self._contexts.pop(session["id"], None)
            raise
        with self._lock:
            binding.expires_at = time.monotonic() + self.ttl_seconds
        return binding.capability

    @contextmanager
    def turn(self, session: dict):
        session_id = session["id"]
        with self._lock:
            self._reap()
            if session_id in self._active or session_id in self._closing:
                raise HTTPException(409, "isolated session is busy or closing")
            self._active.add(session_id)
        try:
            yield self.context_for(session)
        finally:
            with self._lock:
                self._active.discard(session_id)
                binding = self._contexts.get(session_id)
                if binding is not None:
                    binding.expires_at = time.monotonic() + self.ttl_seconds

    def close_context(self, context: str) -> None:
        self._request("DELETE", context=context)

    def close(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._active or session_id in self._closing:
                raise HTTPException(409, "isolated session is busy or closing")
            binding = self._contexts.get(session_id)
            if binding is None:
                return
            self._closing.add(session_id)
        try:
            self.close_context(binding.capability)
            with self._lock:
                self._contexts.pop(session_id, None)
        finally:
            with self._lock:
                self._closing.discard(session_id)
