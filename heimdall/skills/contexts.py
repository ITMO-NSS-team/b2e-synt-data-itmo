"""Private catalog views. Capabilities select a view, never a data permission."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from math import isfinite
import secrets
import threading
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from .registry import Registry, _split_frontmatter


class ExtraSkill(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=262_144)


class CreateCatalog(BaseModel):
    employee_id: str = Field(min_length=1)
    extra_skills: list[ExtraSkill] = Field(min_length=1, max_length=16)


@dataclass
class CatalogView:
    employee_id: str
    registry: Registry
    evidence: dict
    expires_at: float


class CatalogContexts:
    def __init__(self, ttl_seconds: float | None = None) -> None:
        self.ttl_seconds = (ttl_seconds if ttl_seconds is not None else
                            float(os.getenv("HEIMDALL_CATALOG_TTL_SECONDS", "86400")))
        if not isfinite(self.ttl_seconds) or self.ttl_seconds <= 0:
            raise ValueError("catalog TTL must be positive")
        self._views: dict[str, CatalogView] = {}
        self._lock = threading.Lock()

    def _reap(self) -> None:
        now = time.monotonic()
        for key in [key for key, view in self._views.items() if view.expires_at <= now]:
            del self._views[key]

    def create(self, base: Registry, request: CreateCatalog) -> dict:
        if sum(len(file.content.encode()) for file in request.extra_skills) > 1_048_576:
            raise HTTPException(422, "extra skills batch exceeds 1 MB")
        if any(len(file.content.encode()) > 262_144 for file in request.extra_skills):
            raise HTTPException(422, "extra skill exceeds 256 KiB")
        baseline_hash = base.content_hash()
        try:
            registry = base.with_references([(file.filename, file.content)
                                            for file in request.extra_skills])
            catalog_hash = registry.content_hash()
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise HTTPException(422, str(exc)) from exc
        evidence = {
            "baseline_hash": baseline_hash,
            "catalog_hash": catalog_hash,
            "native_skills": [{"filename": file.filename,
                              "name": _split_frontmatter(file.content)[0]["name"],
                              "sha256": sha256(file.content.encode()).hexdigest()}
                             for file in request.extra_skills],
        }
        capability = secrets.token_urlsafe(32)
        with self._lock:
            self._reap()
            self._views[capability] = CatalogView(
                request.employee_id, registry, evidence,
                time.monotonic() + self.ttl_seconds)
        return {"catalog_context": capability, "catalog_evidence": evidence}

    def resolve(self, capability: str, employee_id: str) -> CatalogView:
        with self._lock:
            self._reap()
            view = self._views.get(capability)
            if view is None:
                raise HTTPException(410, "isolated skill catalog missing or expired")
            if view.employee_id != employee_id:
                raise HTTPException(403, "skill catalog belongs to another employee")
            view.expires_at = time.monotonic() + self.ttl_seconds
            return view

    def close(self, capability: str) -> None:
        with self._lock:
            self._reap()
            self._views.pop(capability, None)


def catalog_router(state) -> APIRouter:
    router = APIRouter(prefix="/internal/skill-catalogs", tags=["internal"])
    admin_key = os.getenv("HEIMDALL_CATALOG_ADMIN_KEY", "")

    def authorize(key: str | None) -> None:
        if not admin_key:
            raise HTTPException(503, "isolated catalog provisioning is not configured")
        if not key or not secrets.compare_digest(key.encode(), admin_key.encode()):
            raise HTTPException(403, "invalid catalog service key")

    @router.post("", status_code=201)
    def create(payload: CreateCatalog,
               x_catalog_admin_key: str | None = Header(default=None)) -> dict:
        authorize(x_catalog_admin_key)
        return state.catalog_contexts.create(state.registry, payload)

    @router.get("/current")
    def inspect(employee_id: str, x_skill_catalog_context: str = Header(),
                x_catalog_admin_key: str | None = Header(default=None)) -> dict:
        authorize(x_catalog_admin_key)
        return state.catalog_contexts.resolve(x_skill_catalog_context, employee_id).evidence

    @router.delete("/current", status_code=204)
    def close(x_skill_catalog_context: str = Header(),
              x_catalog_admin_key: str | None = Header(default=None)) -> None:
        authorize(x_catalog_admin_key)
        state.catalog_contexts.close(x_skill_catalog_context)

    return router
