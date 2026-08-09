"""Delivering a memory pack into the agent's prompt.

``_memory_block`` needs a registry to resolve ``reflected`` memory, but nothing
else ``sim.agent.app.AgentState`` carries — a bare stand-in with a ``.registry``
attribute exercises the real code path without paying for harness bootstrap or
the data-small corpus that other agent tests require.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sim.agent.app import _memory_block
from sim.agent.config import AgentConfig
from sim.agent.prompt import DEFAULT_SYSTEM_PROMPT, render
from sim.reflection.memory import MemoryItem, build_pack, commit_pack
from sim.registry import Registry


@pytest.fixture()
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    yield reg
    reg.close()


def _state(registry: Registry):
    return SimpleNamespace(registry=registry)


def _committed_ref(registry: Registry, *, arm: str = "isolated", scope: str = "i1") -> str:
    item = MemoryItem(
        id="m_1", scope=scope, kind="method", trigger={}, text="Запрашивай агрегат.",
        support=3, refute=0, episodes_support=("ep_1",), episodes_refute=(),
        created_epoch=1, last_useful_epoch=1, epochs_idle=0,
        origin_instances=(scope,), tokens=20,
    )
    pack = build_pack(arm, scope, 1, (item,), None)
    return commit_pack(registry, pack, actor="t")


# --------------------------------------------------------------- AgentConfig


def test_reflected_strategy_without_a_memory_ref_is_refused():
    with pytest.raises(ValueError):
        AgentConfig(memory_strategy="reflected", memory_ref="")


def test_a_memory_ref_outside_the_reflected_strategy_is_refused():
    with pytest.raises(ValueError):
        AgentConfig(memory_strategy="none", memory_ref="memory_isolated_i1@1")


def test_an_unpinned_memory_ref_is_refused():
    with pytest.raises(ValueError):
        AgentConfig(memory_strategy="reflected", memory_ref="memory_isolated_i1")


def test_a_pinned_memory_ref_is_accepted():
    config = AgentConfig(memory_strategy="reflected", memory_ref="memory_isolated_i1@3")
    assert config.memory_ref == "memory_isolated_i1@3"


def test_a_config_predating_memory_ref_still_loads():
    # from_dict rejects only unknown keys, so a blob committed before this
    # field existed must still construct with the new field defaulted.
    legacy = AgentConfig().as_dict()
    del legacy["memory_ref"]
    loaded = AgentConfig.from_dict(legacy)
    assert loaded.memory_ref == ""
    assert loaded.memory_strategy == "none"


# ------------------------------------------------------------- _memory_block


def test_memory_block_reflected_returns_the_packs_exact_rendered_string(registry):
    ref = _committed_ref(registry)
    config = AgentConfig(memory_strategy="reflected", memory_ref=ref)
    from sim.reflection.memory import load_pack
    expected = load_pack(registry, ref).rendered

    block = _memory_block(_state(registry), config, {"employee_id": "E1"})
    assert block == expected
    assert block != ""


def test_memory_block_none_strategy_is_still_empty(registry):
    config = AgentConfig(memory_strategy="none")
    assert _memory_block(_state(registry), config, {"employee_id": "E1"}) == ""


def test_memory_block_rq3_strategies_do_not_touch_the_registry(registry):
    # A state whose .registry would raise on any attribute access other than
    # existing, so these branches proving they never touch it.
    registry.close()
    config = AgentConfig(memory_strategy="current_mart")
    block = _memory_block(_state(registry), config, {"employee_id": "E1"})
    assert "E1" in block


# ----------------------------------------------------------------- render()


def test_prompt_still_renders_with_a_memory_block():
    out = render(DEFAULT_SYSTEM_PROMPT, {"employee_id": "1", "memory_block": "текст"})
    assert "текст" in out


def test_prompt_still_renders_without_a_memory_block():
    out = render(DEFAULT_SYSTEM_PROMPT, {"employee_id": "1", "memory_block": ""})
    assert "Что известно о сотруднике" not in out
