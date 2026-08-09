"""The reflection procedure: one pipeline, two inputs.

The load-bearing test in this file is
``test_shared_and_isolated_differ_only_in_input`` — everything else here
exists to make that test's premise trustworthy (the guard really does record
what it rejects, a no-op epoch really does not bump the registry, a partial
epoch really is refused) rather than to explore ``reflect()``'s behaviour
generally.
"""
from __future__ import annotations

import json

import pytest

from sim.reflection import curate, reflect as reflect_mod
from sim.reflection.aggregate import Episode
from sim.reflection.extract import CallRecord
from sim.reflection.memory import MemoryItem
from sim.registry import Registry
from sim.research.evaluate import TraceFacts

reflect = reflect_mod.reflect


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    yield reg
    reg.close()


def _facts(**kw) -> TraceFacts:
    base = dict(http_statuses=(200,), heimdall_calls=1, rows_returned=(3,),
               error_codes=(), repeated_calls=0, pagination_walks=0,
               columns_requested=(3,), tokens=1200, seconds=4.0)
    base.update(kw)
    return TraceFacts(**base)


def _call(seq: int = 0, **kw) -> CallRecord:
    base = dict(tool="mcp_query", schema="dm_core", logic_model="employee_actual",
               columns=("grade_level",), filter_nodes=(), order_by=(),
               limit_bucket="none", http_status=200, error_code=None, rows=3,
               argument_keys=())
    base.update(kw)
    return CallRecord(seq=seq, **base)


def _episode(id: str, *, correct: bool = True, scored: bool = True,
            refused: bool = False, question_class: str = "count_by_grade",
            family: str = "org", category: str = "answerable") -> Episode:
    return Episode(id=id, question_class=question_class, family=family,
                  category=category, calls=(_call(),), facts=_facts(),
                  correct=correct, scored=scored, refused=refused, gold=None)


def _item(*, id: str = "m_fixture", kind: str = "method", support: int = 0,
         refute: int = 0, text: str = "Проверяй схему перед запросом.",
         tokens: int = 20, scope: str = "i1", trigger: dict | None = None,
         episodes_support: tuple = (), episodes_refute: tuple = (),
         created_epoch: int = 1, last_useful_epoch: int = 1, epochs_idle: int = 0,
         origin_instances: tuple = ()) -> MemoryItem:
    return MemoryItem(
        id=id, scope=scope, kind=kind, trigger=trigger or {}, text=text,
        support=support, refute=refute, episodes_support=tuple(episodes_support),
        episodes_refute=tuple(episodes_refute), created_epoch=created_epoch,
        last_useful_epoch=last_useful_epoch, epochs_idle=epochs_idle,
        origin_instances=tuple(origin_instances), tokens=tokens,
    )


class _FakeClient:
    """Records every prompt it was called with; returns a fixed response."""

    def __init__(self, response: str = "[]") -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _ops_json(*ops: dict) -> str:
    return json.dumps(list(ops), ensure_ascii=False)


# ----------------------------------------------------- the model owns no counter


def test_the_model_cannot_write_its_own_evidence():
    """A curator that both discovers a pattern and counts it will assert
    'this fails often' from two episodes. Counters are owned by code."""
    ops = curate.parse_ops('[{"op":"ADD","kind":"method","trigger":{},'
                           '"text":"x","support":99}]')
    assert len(ops) == 1
    assert all(not hasattr(o, "support") and not hasattr(o, "refute") for o in ops)


def test_parse_ops_drops_a_malformed_entry_but_keeps_the_rest():
    raw = _ops_json(
        {"op": "ADD", "kind": "bogus_kind", "trigger": {}, "text": "x"},
        {"op": "ADD", "kind": "method", "trigger": {}, "text": "Годная формулировка."},
    )
    ops = curate.parse_ops(raw)
    assert len(ops) == 1
    assert ops[0].text == "Годная формулировка."


def test_parse_ops_recovers_json_from_inside_a_markdown_fence():
    raw = "Вот операции:\n```json\n" + _ops_json(
        {"op": "DROP", "id": "m_1", "reason": "устарело"}) + "\n```\nГотово."
    ops = curate.parse_ops(raw)
    assert len(ops) == 1
    assert ops[0].id == "m_1"


def test_parse_ops_caps_at_eight_even_when_the_model_sends_more():
    raw = _ops_json(*[
        {"op": "ADD", "kind": "method", "trigger": {}, "text": f"Урок номер {i}."}
        for i in range(20)
    ])
    ops = curate.parse_ops(raw)
    assert len(ops) == curate.MAX_OPS


# -------------------------------------------------------------- shared merge


def test_shared_merge_unions_episodes_rather_than_summing_them():
    """Three instances that saw the SAME episode must count once. Summing is
    the 3x-data confound arriving through the counter."""
    a = _item(id="m_shared", support=3, refute=0,
             episodes_support=("ep_1", "ep_2", "ep_3"), origin_instances=("i1",))
    b = _item(id="m_shared", support=2, refute=0,
             episodes_support=("ep_2", "ep_3"), origin_instances=("i2",))

    merged = curate.merge_shared([a, b])

    assert len(merged) == 1
    assert set(merged[0].episodes_support) == {"ep_1", "ep_2", "ep_3"}
    naive_sum = a.support + b.support
    assert merged[0].support != naive_sum
    assert merged[0].support == len({"ep_1", "ep_2", "ep_3"}) + 1  # cross-instance bonus
    assert set(merged[0].origin_instances) == {"i1", "i2"}


def test_shared_merge_leaves_a_single_instance_item_without_a_bonus():
    a = _item(id="m_solo", support=2, episodes_support=("ep_1", "ep_2"),
             origin_instances=("i1",))
    merged = curate.merge_shared([a])
    assert merged[0].support == 2
    assert merged[0].origin_instances == ("i1",)


def test_shared_merge_never_collides_items_with_different_wording():
    """Exact-key merge, not similarity: two candidates for the same trigger
    but different text must NOT merge, since that is exactly the
    embedding-style merge the design spec rules out for this corpus."""
    a = _item(id="m_a", text="Формулировка один.")
    b = _item(id="m_b", text="Формулировка два.")
    merged = curate.merge_shared([a, b])
    assert len(merged) == 2


# --------------------------------------------------- differ only in input


def test_shared_and_isolated_differ_only_in_input(registry):
    """Same episodes, same prior memory, same fake client: reflect(scope='fleet')
    and reflect(scope='i1') must produce identical packs. If they differ, the
    shared arm has a second treatment nobody declared."""
    episodes = [_episode(f"ep_{i}") for i in range(6)]
    client = _FakeClient(_ops_json(
        {"op": "ADD", "kind": "method", "trigger": {"question_class": ["count_by_grade"]},
         "text": "Запрашивай агрегат, а не построчный список."},
    ))

    pack_i1 = reflect(episodes, (), arm="isolated", scope="i1", epoch=1,
                      client=client, registry=registry, epoch_size=6)
    pack_fleet = reflect(episodes, (), arm="shared", scope="fleet", epoch=1,
                         client=client, registry=registry, epoch_size=6)

    assert pack_i1.rendered == pack_fleet.rendered
    assert pack_i1.digest == pack_fleet.digest

    def _content(items):
        return sorted(
            (it.id, it.kind, json.dumps(it.trigger, sort_keys=True), it.text,
             it.support, it.refute, it.episodes_support, it.episodes_refute)
            for it in items
        )

    assert _content(pack_i1.items) == _content(pack_fleet.items)
    # The only place the two calls are allowed to differ is the label.
    assert pack_i1.scope != pack_fleet.scope


# -------------------------------------------------------- no-op night, no bump


def test_a_night_that_learns_nothing_does_not_bump_the_version(registry):
    episodes = [_episode(f"ep_{i}") for i in range(3)]
    client = _FakeClient("[]")

    reflect(episodes, (), arm="isolated", scope="i1", epoch=1, client=client,
           registry=registry, epoch_size=3)
    head_after_first = registry.head("memory_isolated_i1")
    assert head_after_first is not None
    assert head_after_first.version == 1

    reflect(episodes, (), arm="isolated", scope="i1", epoch=2, client=client,
           registry=registry, epoch_size=3)
    head_after_second = registry.head("memory_isolated_i1")

    assert head_after_second.version == head_after_first.version


def test_a_night_that_actually_adds_a_lesson_does_bump_the_version(registry):
    """The counterpart to the no-op test: a real change must still land, or
    the fix for "no-op nights don't bump" would have been to stop reflection
    from ever writing anything."""
    episodes = [_episode(f"ep_{i}") for i in range(3)]
    empty_client = _FakeClient("[]")
    reflect(episodes, (), arm="isolated", scope="i1", epoch=1, client=empty_client,
           registry=registry, epoch_size=3)
    v1 = registry.head("memory_isolated_i1").version

    adding_client = _FakeClient(_ops_json(
        {"op": "ADD", "kind": "method", "trigger": {}, "text": "Новый приём про агрегаты."}))
    reflect(episodes, (), arm="isolated", scope="i1", epoch=2, client=adding_client,
           registry=registry, epoch_size=3)
    v2 = registry.head("memory_isolated_i1").version

    assert v2 != v1


# --------------------------------------------------------------- mid-epoch


def test_reflection_refuses_to_run_mid_epoch(registry):
    episodes = [_episode(f"ep_{i}") for i in range(4)]
    with pytest.raises(ValueError):
        reflect(episodes, (), arm="isolated", scope="i1", epoch=1,
               client=_FakeClient("[]"), registry=registry, epoch_size=10)


# ------------------------------------------------------ guard rejections logged


def test_guard_rejections_are_recorded_not_silently_dropped(registry):
    episodes = [_episode(f"ep_{i}") for i in range(2)]
    uuid_text = "Если person_id похож на 8f14e45f-ceea-467a-9d0f-2b4c3a1e5b77, не выдумывай остальное."
    client = _FakeClient(_ops_json(
        {"op": "ADD", "kind": "pitfall", "trigger": {}, "text": uuid_text},
        {"op": "ADD", "kind": "method", "trigger": {}, "text": "Годная формулировка про агрегаты."},
    ))

    reflect(episodes, (), arm="isolated", scope="i1", epoch=1, client=client,
           registry=registry, epoch_size=2)

    log = reflect_mod.load_log(registry, "isolated", "i1")
    assert log["guard_rejections"]["G1"] == 1
    assert log["guard_rejection_rate"]["G1"] > 0
    assert log["texts_checked"] == 2
    assert log["ops_applied"] == 1
    assert any(entry["rule"] == "G1" for entry in log["rejected"])


def test_guard_rejection_rate_is_zero_for_a_rule_that_never_fired(registry):
    episodes = [_episode(f"ep_{i}") for i in range(2)]
    client = _FakeClient(_ops_json(
        {"op": "ADD", "kind": "method", "trigger": {}, "text": "Годная формулировка."}))
    reflect(episodes, (), arm="isolated", scope="i1", epoch=1, client=client,
           registry=registry, epoch_size=2)
    log = reflect_mod.load_log(registry, "isolated", "i1")
    assert log["guard_rejection_rate"]["G3"] == 0.0


# ---------------------------------------------------------------- counters


def test_counters_increment_from_episodes_that_matched_and_followed_the_trigger(registry):
    item = _item(id="m_headcount", kind="method", support=0, refute=0,
                trigger={"question_class": ["count_by_grade"]}, scope="i1")
    episodes = [
        _episode("ep_pass", correct=True, question_class="count_by_grade"),
        _episode("ep_fail", correct=False, question_class="count_by_grade"),
        _episode("ep_other", correct=True, question_class="lookup_grade"),
    ]
    pack = reflect(episodes, (item,), arm="isolated", scope="i1", epoch=2,
                  client=_FakeClient("[]"), registry=registry, epoch_size=3)

    updated = next(it for it in pack.items if it.id == "m_headcount")
    assert updated.support == 1
    assert updated.refute == 1
    assert set(updated.episodes_support) == {"ep_pass"}
    assert set(updated.episodes_refute) == {"ep_fail"}


def test_idle_items_are_retired_after_the_configured_number_of_epochs(registry):
    from sim.reflection.memory import IDLE_RETIRE

    item = _item(id="m_idle", trigger={"question_class": ["never_matches"]}, scope="i1")
    prior = (item,)
    episodes = [_episode("ep_x", question_class="count_by_grade")]

    epoch = 1
    for _ in range(IDLE_RETIRE):
        pack = reflect(episodes, prior, arm="isolated", scope="i1", epoch=epoch,
                       client=_FakeClient("[]"), registry=registry, epoch_size=1)
        prior = pack.items
        epoch += 1

    assert all(it.id != "m_idle" for it in pack.items)


# ------------------------------------------------------------- caching client


def test_caching_curator_client_serves_a_cached_response_without_calling_the_inner_client(registry):
    calls = {"n": 0}

    class _CountingClient:
        def complete(self, prompt: str) -> str:
            calls["n"] += 1
            return "[]"

    caching = curate.CachingCuratorClient(inner=_CountingClient(), registry=registry)
    first = caching.complete("same prompt")
    second = caching.complete("same prompt")

    assert first == second == "[]"
    assert calls["n"] == 1
