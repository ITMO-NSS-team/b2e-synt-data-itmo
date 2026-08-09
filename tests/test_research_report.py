"""``sim/research/report.py``, exercised against synthetic ``turns.jsonl``
rows — never a live run. Covers the six report sections in the order the
plan requires them (health, guard, primary, learning curve, cost, limits),
the statistical machinery (permutation test, bootstrap CI) against
known-answer synthetic data, and the partial-run robustness properties: a
missing file, a truncated trailing line, a missing arm, and a not-yet-complete
epoch must all produce a sensible report rather than a crash.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.research import report as r  # noqa: E402

ARMS = ("A1", "A2", "A3")


# --------------------------------------------------------------------- rows


def _row(*, arm: str, epoch: int, instance: int, qid: str, category: str = "answerable",
        scored: bool = True, correct: bool | None = True, dispatch_failed: bool = False,
        replication: int = 1, tokens: int = 1000, seconds: float = 1.0,
        wall_seconds: float = 1.0, heimdall_calls: int = 2, memory_tokens: int = 0,
        answer_ids: list[str] | None = None, score_reason: str = "",
        family: str = "org", question_class: str = "count") -> dict[str, Any]:
    return {
        "replication": replication, "arm": arm, "epoch": epoch, "instance": instance,
        "question_id": qid, "pair_id": qid, "category": category, "family": family,
        "question_class": question_class, "employee_id": "e1", "config_ref": "c@1",
        "dispatched_at": 0.0, "attempts": 1, "dispatch_failed": dispatch_failed,
        "dispatch_reason": None, "session_id": "s", "trace_id": "t",
        "wall_seconds": wall_seconds, "scored": scored, "correct": correct,
        "score_reason": score_reason, "answer_present": True, "answer_verdict": None,
        "answer_ids": answer_ids or [], "answer_value": None, "answer_refused": False,
        "answer_reason": None, "answer_field_errors": {}, "tokens": tokens,
        "seconds": seconds, "heimdall_calls": heimdall_calls,
        "memory_tokens": memory_tokens, "dry_run": True,
    }


def _full_epoch(arm: str, epoch: int, *, correct: bool = True, category: str = "answerable",
                replication: int = 1, tokens: int = 1000, memory_tokens: int = 0) -> list[dict]:
    """30 rows — one arm's full epoch (10 per instance x 3 instances), the
    unit ``complete_epochs`` requires."""
    out = []
    for instance in range(1, 4):
        for i in range(10):
            qid = f"epoch{epoch}-inst{instance}-q{i}"
            out.append(_row(arm=arm, epoch=epoch, instance=instance, qid=qid,
                            category=category, correct=correct, replication=replication,
                            tokens=tokens, memory_tokens=memory_tokens))
    return out


# ------------------------------------------------------------------- load


def test_load_turns_missing_file_returns_empty(tmp_path: Path) -> None:
    result = r.load_turns(tmp_path / "nope.jsonl")
    assert result.rows == ()
    assert result.total_lines == 0
    assert result.skipped_lines == 0


def test_load_turns_skips_blank_and_truncated_lines(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    good = _row(arm="A1", epoch=0, instance=1, qid="q1")
    path.write_text(json.dumps(good) + "\n\n" + '{"arm": "A2", truncated' , encoding="utf-8")
    result = r.load_turns(path)
    assert len(result.rows) == 1
    assert result.rows[0]["question_id"] == "q1"
    assert result.skipped_lines == 1


def test_dedup_latest_keeps_last_row_for_a_repeated_key() -> None:
    first = _row(arm="A1", epoch=0, instance=1, qid="q1", correct=True)
    second = _row(arm="A1", epoch=0, instance=1, qid="q1", correct=False)
    out = r._dedup_latest((first, second))
    assert len(out) == 1
    assert out[0]["correct"] is False


# ------------------------------------------------------------------ health


def test_health_counts_dispatch_failed_and_scored_false_reasons() -> None:
    rows = (
        _row(arm="A1", epoch=0, instance=1, qid="q1", scored=True, correct=True),
        _row(arm="A1", epoch=0, instance=1, qid="q2", scored=False, correct=None,
            score_reason="no_answer_block"),
        _row(arm="A1", epoch=0, instance=1, qid="q3", dispatch_failed=True, scored=False,
            correct=None, score_reason="dispatch_failed"),
    )
    health = r.health_by_arm(rows)
    a1 = health["A1"]
    assert a1.total == 3
    assert a1.dispatch_failed == 1
    assert a1.scored == 1
    assert a1.scored_false == 2
    assert a1.scored_false_rate == pytest.approx(2 / 3)
    assert a1.reasons == {"no_answer_block": 1, "dispatch_failed": 1}
    # Arms with zero rows still appear, with a null rate rather than a
    # divide-by-zero or a silently-omitted row.
    assert health["A2"].total == 0
    assert health["A2"].scored_false_rate is None


def test_health_warning_fires_above_threshold_and_not_below() -> None:
    # A1 all scored, A2 half scored: a 50pp spread must trip the warning.
    rows = tuple(
        [_row(arm="A1", epoch=0, instance=1, qid=f"a{i}") for i in range(10)]
        + [_row(arm="A2", epoch=0, instance=1, qid=f"b{i}", scored=(i < 5),
                correct=True if i < 5 else None,
                score_reason="" if i < 5 else "no_answer_block")
          for i in range(10)]
    )
    health = r.health_by_arm(rows)
    warning = r.health_warning(health)
    assert warning is not None
    assert "differs by" in warning

    # Same scored fraction on both arms present: no warning.
    close_rows = tuple(
        [_row(arm="A1", epoch=0, instance=1, qid=f"a{i}") for i in range(10)]
        + [_row(arm="A2", epoch=0, instance=1, qid=f"b{i}") for i in range(10)]
    )
    assert r.health_warning(r.health_by_arm(close_rows)) is None


# ------------------------------------------------------------------- guard


def test_guard_caution_pass_rate_excludes_answerable_rows() -> None:
    rows = (
        _row(arm="A1", epoch=0, instance=1, qid="q1", category="answerable", correct=False),
        _row(arm="A1", epoch=0, instance=1, qid="q2", category="no_data", correct=True),
        _row(arm="A1", epoch=0, instance=1, qid="q3", category="out_of_scope", correct=False),
    )
    guard = r.guard_by_arm(rows, valid_ids=None)
    a1 = guard["A1"]
    assert a1.caution_n == 2
    assert a1.caution_pass_rate == pytest.approx(0.5)  # 1 of 2 caution rows correct


def test_guard_followed_injection_rate_is_scored_and_incorrect() -> None:
    rows = (
        # obeyed: scored True, correct False
        _row(arm="A1", epoch=0, instance=1, qid="q1", category="prompt_injection",
            scored=True, correct=False),
        # declined correctly: scored True, correct True
        _row(arm="A1", epoch=0, instance=1, qid="q2", category="prompt_injection",
            scored=True, correct=True),
        # unscored: no answer block at all — not counted as "followed"
        _row(arm="A1", epoch=0, instance=1, qid="q3", category="prompt_injection",
            scored=False, correct=None, score_reason="no_answer_block"),
    )
    guard = r.guard_by_arm(rows, valid_ids=None)
    assert guard["A1"].followed_injection_rate == pytest.approx(1 / 3)


def test_guard_fabricated_id_rate_flags_unknown_ids() -> None:
    rows = (
        _row(arm="A1", epoch=0, instance=1, qid="q1", answer_ids=["real-1"]),
        _row(arm="A1", epoch=0, instance=1, qid="q2", answer_ids=["fake-9"]),
        _row(arm="A1", epoch=0, instance=1, qid="q3", answer_ids=[]),  # no ids: excluded
    )
    guard = r.guard_by_arm(rows, valid_ids=frozenset({"real-1"}))
    a1 = guard["A1"]
    assert a1.ids_seen == 2
    assert a1.fabricated_id_rate == pytest.approx(0.5)
    assert a1.fabricated_id_rate_note is None


def test_guard_fabricated_id_rate_unavailable_without_snapshot() -> None:
    rows = (_row(arm="A1", epoch=0, instance=1, qid="q1", answer_ids=["x"]),)
    guard = r.guard_by_arm(rows, valid_ids=None)
    assert guard["A1"].fabricated_id_rate is None
    assert "unavailable" in guard["A1"].fabricated_id_rate_note


def test_trade_off_warning_fires_only_when_both_halves_move_opposite_ways() -> None:
    # A1: half-and-half on both halves.
    a1_rows = (
        _row(arm="A1", epoch=0, instance=1, qid="d1", category="answerable", correct=True),
        _row(arm="A1", epoch=0, instance=1, qid="d2", category="answerable", correct=False),
        _row(arm="A1", epoch=0, instance=1, qid="c1", category="no_data", correct=True),
        _row(arm="A1", epoch=0, instance=1, qid="c2", category="no_data", correct=False),
    )
    # A2: gains on deterministic (both correct), loses on caution (both wrong).
    a2_traded = (
        _row(arm="A2", epoch=0, instance=1, qid="d1", category="answerable", correct=True),
        _row(arm="A2", epoch=0, instance=1, qid="d2", category="answerable", correct=True),
        _row(arm="A2", epoch=0, instance=1, qid="c1", category="no_data", correct=False),
        _row(arm="A2", epoch=0, instance=1, qid="c2", category="no_data", correct=False),
    )
    rows = a1_rows + a2_traded
    guard = r.guard_by_arm(rows, valid_ids=None)
    warnings = r.trade_off_warnings(rows, guard)
    assert any("A2" in w and "traded" in w for w in warnings)

    # A3: gains on both halves — never a trade, never a warning.
    a3_gains_both = (
        _row(arm="A3", epoch=0, instance=1, qid="d1", category="answerable", correct=True),
        _row(arm="A3", epoch=0, instance=1, qid="d2", category="answerable", correct=True),
        _row(arm="A3", epoch=0, instance=1, qid="c1", category="no_data", correct=True),
        _row(arm="A3", epoch=0, instance=1, qid="c2", category="no_data", correct=True),
    )
    rows2 = a1_rows + a3_gains_both
    guard2 = r.guard_by_arm(rows2, valid_ids=None)
    warnings2 = r.trade_off_warnings(rows2, guard2)
    assert not any("A3" in w for w in warnings2)


# ------------------------------------------------------------------ epochs


def test_complete_epochs_requires_all_three_arms_at_full_count() -> None:
    rows = tuple(_full_epoch("A1", 0) + _full_epoch("A2", 0) + _full_epoch("A3", 0)
                + _full_epoch("A1", 1) + _full_epoch("A2", 1))  # A3 epoch 1 missing
    assert r.complete_epochs(rows) == (0,)
    assert r.epochs_found(rows) == (0, 1)


# -------------------------------------------------------------- paired stats


def test_permutation_and_bootstrap_detect_a_real_constant_gap() -> None:
    rng = np.random.default_rng(1)
    diffs = np.full(60, 1.0)  # A2 always beats A1 on every pair
    p = r.permutation_test(diffs, rng=rng)
    lo, hi = r.bootstrap_ci(diffs, rng=rng)
    assert p is not None and p < 0.01
    assert lo is not None and hi is not None
    assert lo > 0  # CI entirely above zero: a real, positive effect


def test_permutation_test_is_not_significant_for_pure_noise() -> None:
    rng = np.random.default_rng(2)
    # Balanced +1/-1 noise around zero: no consistent winner.
    diffs = np.array(([1.0, -1.0] * 30), dtype=float)
    p = r.permutation_test(diffs, rng=rng)
    assert p is not None and p > 0.5


def test_paired_diffs_only_uses_pairs_both_arms_scored() -> None:
    rows = (
        _row(arm="A1", epoch=0, instance=1, qid="q1", correct=True),
        _row(arm="A2", epoch=0, instance=1, qid="q1", correct=False),
        _row(arm="A1", epoch=0, instance=1, qid="q2", correct=True),
        # A2's q2 is unscored: this pair must be dropped entirely.
        _row(arm="A2", epoch=0, instance=1, qid="q2", scored=False, correct=None),
    )
    diffs = r.paired_diffs(rows, epoch=0, arm="A2", baseline="A1")
    assert diffs.tolist() == [-1.0]


def test_pass_rate_excludes_unscored_rows_from_denominator() -> None:
    rows = (
        _row(arm="A1", epoch=0, instance=1, qid="q1", scored=True, correct=True),
        _row(arm="A1", epoch=0, instance=1, qid="q2", scored=True, correct=False),
        _row(arm="A1", epoch=0, instance=1, qid="q3", scored=False, correct=None),
    )
    rate, n_correct, n_scored = r.pass_rate(rows, arm="A1", epoch=0)
    assert n_scored == 2
    assert n_correct == 1
    assert rate == pytest.approx(0.5)


# -------------------------------------------------------------------- cost


def test_cost_per_correct_excludes_dispatch_failed_and_normalises_by_correct() -> None:
    rows = (
        _row(arm="A1", epoch=0, instance=1, qid="q1", correct=True, tokens=1000,
            wall_seconds=2.0, seconds=1.0, heimdall_calls=3, memory_tokens=0),
        _row(arm="A1", epoch=0, instance=1, qid="q2", correct=False, tokens=500,
            wall_seconds=1.0, seconds=0.5, heimdall_calls=1, memory_tokens=0),
        _row(arm="A1", epoch=0, instance=1, qid="q3", dispatch_failed=True, scored=False,
            correct=None, tokens=0, wall_seconds=0.0, seconds=0.0, heimdall_calls=0),
    )
    cost = r.cost_by_arm(rows)["A1"]
    assert cost.n_correct == 1
    # tokens summed over every non-dispatch-failed row (1000+500), divided by
    # the single correct answer — cost of *everything it took*, not just the
    # correct turn's own tokens.
    assert cost.tokens_per_correct == pytest.approx(1500.0)
    assert cost.wall_seconds_per_correct == pytest.approx(3.0)
    assert cost.heimdall_calls_per_answer == pytest.approx((3 + 1) / 2)


def test_cost_memory_tokens_grouped_by_epoch() -> None:
    rows = (
        _row(arm="A2", epoch=0, instance=1, qid="q1", memory_tokens=100),
        _row(arm="A2", epoch=0, instance=1, qid="q2", memory_tokens=300),
        _row(arm="A2", epoch=1, instance=1, qid="q3", memory_tokens=500),
    )
    cost = r.cost_by_arm(rows)["A2"]
    assert cost.memory_tokens_by_epoch == {0: 200.0, 1: 500.0}


# --------------------------------------------------------------- learning


def test_learning_curve_slope_is_positive_for_a_rising_gap() -> None:
    rows: list[dict] = []
    for epoch in range(3):
        rows += _full_epoch("A1", epoch, correct=True)
        # A2's pass rate rises: more correct each epoch than A1.
        n_correct = 15 + epoch * 5  # 15, 20, 25 of 30
        a2 = _full_epoch("A2", epoch, correct=True)
        for i, row in enumerate(a2):
            row["correct"] = i < n_correct
        rows += a2
        rows += _full_epoch("A3", epoch, correct=True)  # flat, matches A1
    rng = np.random.default_rng(0)
    curve = r.learning_curve(tuple(rows), (0, 1, 2), rng)
    assert curve.slope_a2_pp_per_epoch is not None
    assert curve.slope_a2_pp_per_epoch > 0
    assert curve.slope_a3_pp_per_epoch == pytest.approx(0.0, abs=1e-9)


def test_learning_curve_slope_is_none_with_fewer_than_two_epochs() -> None:
    rng = np.random.default_rng(0)
    rows = tuple(_full_epoch("A1", 0) + _full_epoch("A2", 0) + _full_epoch("A3", 0))
    curve = r.learning_curve(rows, (0,), rng)
    assert curve.slope_a2_pp_per_epoch is None
    assert curve.slope_a3_pp_per_epoch is None


# --------------------------------------------------------------- replication


def test_select_replication_picks_highest_and_names_the_rest() -> None:
    rows = (
        _row(arm="A1", epoch=0, instance=1, qid="q1", replication=1),
        _row(arm="A1", epoch=0, instance=1, qid="q2", replication=2),
    )
    used, others = r._select_replication(rows)
    assert used == 2
    assert others == (1,)


# ------------------------------------------------------------------ report


def _complete_dataset() -> tuple[dict, ...]:
    rows: list[dict] = []
    for epoch in range(2):
        rows += _full_epoch("A1", epoch, correct=True)
        rows += _full_epoch("A2", epoch, correct=True)
        rows += _full_epoch("A3", epoch, correct=True)
    return tuple(rows)


def test_build_report_end_to_end_with_complete_data() -> None:
    report = r.build_report(_complete_dataset(), run_id="t1", total_lines=180,
                            skipped_lines=0, valid_ids=None, seed=0)
    assert report.epochs_complete == (0, 1)
    assert report.final_epoch == 1
    assert report.health_warning is None
    assert report.primary["A2"].mean_diff_pp == pytest.approx(0.0)
    md = r.render_markdown(report)
    for heading in ("## 1. Health", "## 2. Guard", "## 3. Primary",
                    "## 4. Learning curve", "## 5. Cost", "## 6. Honest limits"):
        assert heading in md
    assert "One replication" in md
    payload = r.to_json(report)
    json.dumps(payload)  # round-trips through the stdlib encoder without error


def test_build_report_never_crashes_on_missing_arm() -> None:
    rows = tuple(_full_epoch("A1", 0))  # A2, A3 entirely absent
    report = r.build_report(rows, run_id="t2", total_lines=30, skipped_lines=0,
                            valid_ids=None, seed=0)
    assert report.final_epoch is None  # not complete: A2/A3 never showed up
    assert report.epochs_found == (0,)
    md = r.render_markdown(report)
    assert "No epoch is complete" in md
    json.dumps(r.to_json(report))


def test_build_report_never_crashes_on_empty_input() -> None:
    report = r.build_report((), run_id="t3", total_lines=0, skipped_lines=0,
                            valid_ids=None, seed=0)
    assert report.final_epoch is None
    assert report.n_rows == 0
    md = r.render_markdown(report)
    assert "0 of" in md
    json.dumps(r.to_json(report))


# ---------------------------------------------------------------------- CLI


def test_main_writes_summary_json_and_md_beside_turns(tmp_path: Path) -> None:
    out_dir = tmp_path / "rq4" / "runX"
    out_dir.mkdir(parents=True)
    rows = _complete_dataset()
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    r.main(["--run-id", "runX", "--out-dir", str(out_dir), "--skip-fabricated-id-check"])

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "runX"
    assert payload["final_epoch"] == 1


def test_main_handles_a_missing_turns_file_without_crashing(tmp_path: Path) -> None:
    out_dir = tmp_path / "rq4" / "runY"
    r.main(["--run-id", "runY", "--out-dir", str(out_dir), "--skip-fabricated-id-check"])
    assert (out_dir / "summary.md").exists()
    md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "0 of" in md
