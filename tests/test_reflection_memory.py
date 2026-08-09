"""The memory artefact: score, budget, deterministic renderer, storage.

No LLM anywhere in this file — everything ``sim.reflection.memory`` does is a
pure function or a registry read/write, so these tests build ``MemoryItem``
values by hand rather than going through reflection (Task 5).
"""
from __future__ import annotations

import pytest

from sim.agent.loop import estimate_tokens
from sim.reflection import memory
from sim.registry import Registry

_seq = iter(range(1_000_000))


def _item(*, id: str | None = None, kind: str = "method", support: int = 0,
         refute: int = 0, text: str = "Проверяй схему перед агрегирующим запросом.",
         tokens: int = 20, scope: str = "i1", trigger: dict | None = None,
         episodes_support: tuple = (), episodes_refute: tuple = (),
         created_epoch: int = 1, last_useful_epoch: int = 1, epochs_idle: int = 0,
         origin_instances: tuple = ()) -> memory.MemoryItem:
    return memory.MemoryItem(
        id=id or f"m_{next(_seq):06d}", scope=scope, kind=kind,
        trigger=trigger or {}, text=text, support=support, refute=refute,
        episodes_support=tuple(episodes_support),
        episodes_refute=tuple(episodes_refute),
        created_epoch=created_epoch, last_useful_epoch=last_useful_epoch,
        epochs_idle=epochs_idle, origin_instances=tuple(origin_instances),
        tokens=tokens,
    )


def _pack(*, arm: str = "isolated", scope: str = "i1", epoch: int = 1,
          items: tuple = (), parent_digest: str | None = None) -> memory.MemoryPack:
    return memory.build_pack(arm, scope, epoch, items, parent_digest)


@pytest.fixture()
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    yield reg
    reg.close()


# --------------------------------------------------------------------- score


def test_score_is_laplace_so_one_lucky_hit_cannot_outrank_nine_of_ten():
    lucky = _item(support=1, refute=0)
    solid = _item(support=9, refute=1)
    assert memory.score(solid) > memory.score(lucky)


# -------------------------------------------------------------------- select


def test_select_reserves_capacity_for_pitfalls_even_when_they_score_lower():
    # 30 high-scoring api_mechanic items and 2 low-scoring pitfalls. The
    # pitfalls must still appear: a pack that can only say "do more" is a
    # refusal suppressor, which is the failure the reserve exists to prevent.
    mechanics = [
        _item(kind="api_mechanic", support=20, refute=0, tokens=15,
             text=f"Колонка {i} доступна только через агрегирующий запрос номер {i}.")
        for i in range(30)
    ]
    pitfalls = [
        _item(kind="pitfall", support=0, refute=5, tokens=20,
             text=f"Отказ {i}: код 403 по этой ветке не пытайся обходить.")
        for i in range(2)
    ]
    chosen = memory.select(mechanics + pitfalls)
    chosen_pitfalls = [it for it in chosen if it.kind == "pitfall"]
    assert len(chosen_pitfalls) == 2


def test_select_never_exceeds_the_hard_caps():
    items = [
        _item(kind=("pitfall" if i % 10 == 0 else "method"),
             support=i % 5, refute=(i * 3) % 4, tokens=40,
             text=(f"Пункт номер {i}: перед агрегирующим запросом к витрине "
                   f"проверяй список колонок и не запрашивай {{'*'}}."))
        for i in range(100)
    ]
    chosen = memory.select(items)
    assert len(chosen) <= memory.K_MAX
    assert estimate_tokens([{"content": memory.render(chosen)}]) <= memory.T_MAX


# -------------------------------------------------------------------- render


def test_render_is_deterministic_for_the_same_items():
    items = [
        _item(kind="method", support=3, refute=1,
             text="Запрашивай агрегат, а не построчный список."),
        _item(kind="pitfall", support=0, refute=2,
             text="Отказ 403 не пытайся обойти повторным запросом."),
    ]
    assert memory.render(items) == memory.render(items)


def test_render_contains_no_counters_or_identifiers():
    items = [
        _item(id="m_secret_id", kind="pitfall", support=9, refute=1,
             text="Если данных нет в ответе API, так и скажи."),
    ]
    rendered = memory.render(items)
    assert "m_secret_id" not in rendered
    assert "support" not in rendered and "refute" not in rendered
    # The support/refute counts themselves must not leak as bare numbers next
    # to the lesson text either.
    assert "9" not in rendered and "\n1\n" not in rendered


# ---------------------------------------------------------------- registry


def test_commit_pack_refuses_to_write_an_agent_config_name(registry):
    pack = _pack(arm="agent_config_rq4", items=(_item(),))
    with pytest.raises(AssertionError):
        memory.commit_pack(registry, pack, actor="t")


def test_commit_pack_refuses_a_system_prompt_scope(registry):
    pack = _pack(scope="system_prompt_default", items=(_item(),))
    with pytest.raises(AssertionError):
        memory.commit_pack(registry, pack, actor="t")


def test_commit_pack_is_idempotent_for_identical_content(registry):
    # registry.commit returns the existing head unchanged, so a night that
    # learned nothing must not bump the version — otherwise the loop
    # manufactures a condition change out of a no-op.
    pack = _pack(items=(_item(kind="method", support=4, refute=0),))
    ref1 = memory.commit_pack(registry, pack, actor="t")
    ref2 = memory.commit_pack(registry, pack, actor="t")
    assert ref1 == ref2


def test_commit_pack_bumps_the_version_when_content_actually_changes(registry):
    pack1 = _pack(items=(_item(id="m_1", kind="method", support=1, refute=0),))
    pack2 = _pack(items=(_item(id="m_1", kind="method", support=9, refute=0,
                               text="Другой текст урока."),))
    ref1 = memory.commit_pack(registry, pack1, actor="t")
    ref2 = memory.commit_pack(registry, pack2, actor="t")
    assert ref1 != ref2


def test_load_pack_round_trips_every_field(registry):
    items = (
        _item(id="m_a", kind="pitfall", support=1, refute=2, tokens=30,
             episodes_support=("ep_a",), episodes_refute=("ep_b",),
             origin_instances=("i1", "i2"),
             trigger={"question_class": ["headcount_by_unit"]}),
    )
    pack = _pack(arm="shared", scope="fleet", epoch=3, items=items,
                parent_digest="deadbeef")
    ref = memory.commit_pack(registry, pack, actor="t")
    loaded = memory.load_pack(registry, ref)
    assert loaded == pack
