"""Services must work when published under a path prefix.

The proxy strips the prefix (`handle_path /admin/*`), so each app is served at
/admin, /agent or /research while believing it lives at the root. Three separate
things broke on that, all of them invisible to a health check:

* the admin UI emitted `href="/config"`, so every link in its own navigation
  left the app and landed on the proxy's fallback text — only a hand-typed
  /admin/config worked;
* the Swagger page at /agent/docs asked the browser for `/openapi.json`, got
  that same fallback, and reported "the definition does not specify a valid
  version field";
* a bare `/admin` matched no route and answered the fallback with 200, which
  reads as a working page rather than a missing slash.

Every one of them returned a 2xx somewhere, which is why "the route responds"
was not evidence of anything. These tests check the thing that actually
mattered: where the links point.
"""
from __future__ import annotations

import os
import re

import pytest


# --------------------------------------------------------------- admin UI


def test_admin_nav_is_relative():
    """An absolute href cannot be right under both mounts, and the app is not
    told which one it is under."""
    from sim.admin.app import NAV

    for href, label in NAV:
        assert not href.startswith("/"), f"{label} -> {href}"
        assert not href.startswith("http"), f"{label} -> {href}"


def test_admin_overview_link_is_dot_not_empty():
    """An empty href means "reload this page", which would make the Overview
    tab a no-op everywhere except Overview."""
    from sim.admin.app import NAV

    assert NAV[0][0] == "."


def test_admin_source_has_no_root_anchored_links():
    """Covers the links and form actions that are not in NAV — the ones built
    inline in each page, which is where most of them were."""
    import pathlib

    source = pathlib.Path("sim/admin/app.py").read_text("utf-8")
    offenders = re.findall(r'(?:href|action)="/[^"]*"', source)
    assert not offenders, offenders


def test_admin_redirects_are_relative():
    """A 303 to "/config" leaves the mount just as surely as a link does, and
    it happens right after a successful save — so the change lands and the
    researcher is told it did not."""
    import pathlib

    source = pathlib.Path("sim/admin/app.py").read_text("utf-8")
    offenders = re.findall(r'RedirectResponse\(f?"/[^"]*"', source)
    assert not offenders, offenders


# ------------------------------------------------------- FastAPI root_path


@pytest.mark.parametrize("module,prefix", [
    ("sim.agent.app", "/agent"),
    ("sim.research.app", "/research"),
])
def test_root_path_is_read_from_the_environment(monkeypatch, module, prefix):
    """With root_path set, FastAPI advertises {prefix}/openapi.json — which is
    the one URL Swagger needs and the only one it cannot guess."""
    import importlib

    monkeypatch.setenv("B2E_ROOT_PATH", prefix)
    mod = importlib.import_module(module)
    app = mod.create_app()
    assert app.root_path == prefix


@pytest.mark.parametrize("module", ["sim.agent.app", "sim.research.app"])
def test_unmounted_is_still_the_default(monkeypatch, module):
    """Running bare — `make serve`, tests, a direct container call — must not
    require the variable to be set."""
    import importlib

    monkeypatch.delenv("B2E_ROOT_PATH", raising=False)
    mod = importlib.import_module(module)
    assert mod.create_app().root_path == ""


# ------------------------------------------------------------ the proxy


def _caddyfile() -> str:
    import pathlib

    return pathlib.Path("deploy/Caddyfile").read_text("utf-8")


@pytest.mark.parametrize("prefix", ["/admin", "/research", "/agent", "/phoenix"])
def test_bare_prefix_redirects_to_the_slash(prefix):
    """`handle_path /admin/*` does not match a bare /admin. Without this the
    request reaches the catch-all and gets 200 plus a sentence."""
    assert re.search(rf"redir {re.escape(prefix)} {re.escape(prefix)}/ 308",
                     _caddyfile()), prefix


@pytest.mark.parametrize("service,prefix", [
    ("b2e-agent", "/agent"), ("research-api", "/research"),
])
def test_compose_tells_each_service_its_mount(service, prefix):
    """The prefix is knowledge the app cannot derive, so it has to be supplied
    where the mount is decided — next to the proxy route, in compose."""
    import pathlib

    compose = pathlib.Path("deploy/docker-compose.yml").read_text("utf-8")
    assert f"B2E_ROOT_PATH: {prefix}" in compose, service
