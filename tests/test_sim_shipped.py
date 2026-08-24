"""Shipped configs against a read-only registry.

This file exists because of a bug that CI could not have caught. `b2e-agent`
mounts the registry read-only — the threat model requires that the agent uid
cannot reach the approval store — and its `_bootstrap_registry` had been dead
code for as long as the shipped set never changed: every ref already existed, so
no write was ever attempted.

Adding one new shipped config woke it up. The commit it then tried raised
`attempt to write a readonly database` out of `AgentState.__init__`, which takes
the service down at boot. Nothing in the suite noticed, because every test
builds its registry from an empty writable temp file.

So the read-only mount is simulated here explicitly.
"""
from __future__ import annotations

import pytest

from sim.agent.config import AgentConfig
from sim.agent.shipped import (
    INTERACTIVE_CONFIG_REF, audit, bootstrap, bootstrap_if_writable,
    shipped_configs,
)
from sim.registry import Registry


@pytest.fixture()
def registry(tmp_path):
    r = Registry(tmp_path / "registry.db")
    yield r
    r.close()


def _readonly(path):
    """The same file through a read-only connection.

    `Registry(readonly=True)` opens with SQLite's `mode=ro`, which fails writes
    the same way the container's `:ro` mount does — reads succeed, the first
    commit raises OperationalError. The deployed agent actually opens read-write
    and is refused by the filesystem; the exception both paths produce is the
    same one, and it is the exception that matters here.
    """
    return Registry(path, readonly=True)


# ------------------------------------------------------------ bootstrap


def test_bootstrap_creates_every_shipped_config(registry):
    created = bootstrap(registry)
    assert set(created) == set(shipped_configs())
    for ref in shipped_configs():
        assert registry.head(ref) is not None, ref


def test_bootstrap_is_idempotent(registry):
    bootstrap(registry)
    assert bootstrap(registry) == []


def test_bootstrap_never_overwrites_an_operator_edit(registry):
    """A second call must not revert a config someone deliberately changed."""
    bootstrap(registry)
    registry.commit("agent_config", "agent",
                    AgentConfig(temperature=0.7).as_dict(),
                    actor="researcher", note="warmer")
    bootstrap(registry)
    _version, body = registry.load("agent_config")
    assert body["temperature"] == 0.7


def test_the_openrouter_config_is_messages_api_and_capped(registry):
    bootstrap(registry)
    from sim.agent.config import DEFAULT_OPENROUTER_MODEL
    from sim.agent.shipped import OPENROUTER_CONFIG_REF

    _version, body = registry.load(OPENROUTER_CONFIG_REF)
    config = AgentConfig.from_dict(body)
    assert config.harness == "messages_api"
    assert config.model_id == DEFAULT_OPENROUTER_MODEL
    assert config.max_output_tokens == 4096


# -------------------------------------------------------- read-only path


def test_a_readonly_registry_does_not_raise(tmp_path, registry):
    """The regression. This is the agent's situation in the deployed stack, and
    an exception here is a service that will not boot."""
    status = bootstrap_if_writable(_readonly(tmp_path / "registry.db"))
    assert isinstance(status, str) and status


def test_a_readonly_registry_reports_what_is_missing(tmp_path, registry):
    ro = _readonly(tmp_path / "registry.db")
    status = bootstrap_if_writable(ro)
    assert INTERACTIVE_CONFIG_REF in status
    assert "read-only" in status


def test_the_status_tells_an_operator_how_to_fix_it(tmp_path, registry):
    """A health check that says "broken" without saying "run this" costs
    somebody an hour of reading source."""
    ro = _readonly(tmp_path / "registry.db")
    _missing, status = audit(ro)
    assert "admin-ui" in status


def test_a_seeded_registry_reads_clean_when_readonly(tmp_path, registry):
    """The normal steady state: admin-ui seeded it, the agent only reads."""
    bootstrap(registry)
    ro = _readonly(tmp_path / "registry.db")
    missing, status = audit(ro)
    assert missing == []
    assert status == "complete"


def test_a_writable_registry_still_seeds_itself(registry):
    """`make serve` on the host points the same code at a writable file, and a
    developer should not have to run admin-ui to get a working default."""
    status = bootstrap_if_writable(registry)
    assert "seeded" in status
    assert registry.head(INTERACTIVE_CONFIG_REF) is not None
