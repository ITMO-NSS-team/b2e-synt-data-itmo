"""Pre-aggregation, episode cards, and the gold-value boundary.

The load-bearing assertion (plan Task 4, Step 1): an ``EpisodeCard`` must
never carry answer text or a gold value, checked by serialising a card and
looking for the gold value's string form — not by reading the source and
trusting that nothing copies it, because the whole point of the boundary is
that it holds even when a future edit adds a field carelessly.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from sim.reflection.aggregate import Episode, aggregate, build_episode_card
from sim.reflection.extract import CallRecord
from sim.reflection.memory import MemoryItem
from sim.research.evaluate import TraceFacts


def _facts(**kw) -> TraceFacts:
    base = dict(http_statuses=(), heimdall_calls=1, rows_returned=(),
               error_codes=(), repeated_calls=0, pagination_walks=0,
               columns_requested=(), tokens=1000, seconds=5.0)
    base.update(kw)
    return TraceFacts(**base)


def _call(seq: int, *, tool: str = "mcp_query", schema: str = "dm_core",
         logic_model: str = "employee_actual", columns: tuple = (),
         http_status: int | None = 200, error_code: str | None = None,
         rows: int = 1, argument_keys: tuple = ()) -> CallRecord:
    return CallRecord(seq=seq, tool=tool, schema=schema, logic_model=logic_model,
                      columns=columns, filter_nodes=(), order_by=(),
                      limit_bucket="none", http_status=http_status,
                      error_code=error_code, rows=rows, argument_keys=argument_keys)


def _episode(**kw) -> Episode:
    base = dict(id="ep1", question_class="headcount_by_unit", family="org",
               category="answerable", calls=(_call(0),), facts=_facts(),
               correct=True, scored=True, refused=False, gold=None)
    base.update(kw)
    return Episode(**base)


# ---------------------------------------------------------- gold-value boundary


def test_episode_card_never_carries_the_gold_value():
    episode = _episode(id="ep_secret", correct=False, scored=True, refused=False,
                       gold=294118.0)
    aggregated = aggregate([episode])
    card = build_episode_card(episode, aggregated)

    blob = json.dumps(asdict(card), ensure_ascii=False, default=str)
    assert "294118" not in blob
    assert "ep_secret" not in blob


def test_episode_card_never_carries_a_string_gold_value_either():
    episode = _episode(id="ep_verdict", category="access_control",
                       correct=False, scored=True, refused=False,
                       gold="совершенно секретный текст ответа")
    aggregated = aggregate([episode])
    card = build_episode_card(episode, aggregated)

    blob = json.dumps(asdict(card), ensure_ascii=False, default=str)
    assert "секретный" not in blob


# ------------------------------------------------------------------ verdicts


def test_build_episode_card_reports_pass_for_a_correct_episode():
    episode = _episode(correct=True, scored=True)
    card = build_episode_card(episode, aggregate([episode]))
    assert card.verdict == "PASS"
    assert card.failure_class is None


def test_build_episode_card_reports_unscored_without_inventing_a_failure_class():
    episode = _episode(correct=None, scored=False)
    card = build_episode_card(episode, aggregate([episode]))
    assert card.verdict == "UNSCORED"
    assert card.failure_class == "unscored"


def test_build_episode_card_plan_is_value_free_and_shape_only():
    episode = _episode(calls=(
        _call(0, columns=("grade_level", "unit_name")),
        _call(1, tool="mcp_query", columns=(), argument_keys=("metrics", "schema")),
    ))
    card = build_episode_card(episode, aggregate([episode]))
    assert card.plan == (
        "mcp_query(dm_core.employee_actual, cols=2, rowwise)",
        "mcp_query(dm_core.employee_actual, aggregate)",
    )


# --------------------------------------------------------------- per-class


def test_aggregate_computes_per_class_pass_rate_and_cost():
    pass_ep = _episode(id="a", correct=True, scored=True, facts=_facts(tokens=1000))
    fail_ep = _episode(id="b", correct=False, scored=True, facts=_facts(tokens=2000))
    result = aggregate([pass_ep, fail_ep])
    stats = result["per_class"]["headcount_by_unit"]
    assert stats["n"] == 2
    assert stats["pass_rate"] == 0.5
    assert stats["mean_tokens"] == 1500.0


# --------------------------------------------------------- error histogram


def test_aggregate_reports_the_fix_diff_to_the_next_succeeding_call():
    failing = _call(0, columns=("grade_lvl",), http_status=400,
                    error_code="unknown-column",
                    argument_keys=("columns", "logic_model", "schema"))
    fixed = _call(1, columns=("grade_level",), http_status=200, rows=3,
                 argument_keys=("columns", "limit", "logic_model", "schema"))
    episode = _episode(calls=(failing, fixed))
    result = aggregate([episode])

    key = [k for k in result["api_errors"] if "unknown-column" in k]
    assert len(key) == 1
    entry = result["api_errors"][key[0]]
    assert entry["count"] == 1
    assert entry["fixed_by"] == {"added": ["limit"], "removed": []}


def test_aggregate_reports_no_fix_when_the_error_was_never_recovered_from():
    failing = _call(0, http_status=403, error_code="forbidden", rows=0)
    episode = _episode(calls=(failing,), correct=False, scored=True)
    result = aggregate([episode])
    key = [k for k in result["api_errors"] if "forbidden" in k]
    assert result["api_errors"][key[0]]["fixed_by"] is None


# -------------------------------------------------------------------- waste


def test_aggregate_sums_repeated_calls_and_pagination_walks_across_episodes():
    ep1 = _episode(id="a", facts=_facts(repeated_calls=2, pagination_walks=1))
    ep2 = _episode(id="b", facts=_facts(repeated_calls=1, pagination_walks=0))
    result = aggregate([ep1, ep2])
    assert result["waste"]["repeated_calls"] == 3
    assert result["waste"]["pagination_walks"] == 1


# --------------------------------------------------------------- memory audit


def test_aggregate_memory_audit_matches_episodes_by_trigger_and_scores_them():
    item = MemoryItem(
        id="m1", scope="i1", kind="pitfall",
        trigger={"question_class": ["salary_lookup"]},
        text="Жди 403 на прямой запрос зарплаты одного человека.",
        support=0, refute=0, episodes_support=(), episodes_refute=(),
        created_epoch=1, last_useful_epoch=1, epochs_idle=0,
        origin_instances=(), tokens=15,
    )
    matching = _episode(id="salary_ep", question_class="salary_lookup",
                        category="access_control", correct=True, scored=True,
                        refused=True,
                        calls=(_call(0, http_status=403, error_code="forbidden", rows=0),))
    other = _episode(id="other_ep", question_class="headcount_by_unit")

    result = aggregate([matching, other], memory=[item])
    audit = result["memory_audit"]["m1"]
    assert audit["episodes_matched"] == ("salary_ep",)
    assert audit["episodes_followed"] == ("salary_ep",)
    assert audit["pass_rate_when_followed"] == 1.0


def test_aggregate_memory_audit_covers_every_item_even_with_no_match():
    item = MemoryItem(
        id="m_unused", scope="i1", kind="method",
        trigger={"question_class": ["nonexistent_class"]}, text="…",
        support=0, refute=0, episodes_support=(), episodes_refute=(),
        created_epoch=1, last_useful_epoch=1, epochs_idle=0,
        origin_instances=(), tokens=10,
    )
    result = aggregate([_episode()], memory=[item])
    audit = result["memory_audit"]["m_unused"]
    assert audit["episodes_matched"] == ()
    assert audit["pass_rate_when_followed"] is None
