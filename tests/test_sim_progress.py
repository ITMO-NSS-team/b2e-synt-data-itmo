"""Live turn progress: event folding and the board that holds it.

Pure unit tests — no CLI, no HTTP. The events are real stream-json shapes,
copied from what Claude Code 2.1.220 actually emits, because the whole value of
this module is that it reads those correctly and a hand-invented shape would
prove nothing.
"""
from __future__ import annotations

import time

from sim.agent.progress import (
    MCP_PREFIX, TRAIL, ProgressBoard, TurnProgress, label_for,
)


def _tool_use(name: str) -> dict:
    return {"type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "x", "name": name,
                                     "input": {}}]}}


RESULT = {"type": "result", "is_error": False, "result": "ответ"}


# --------------------------------------------------------------- labelling


def test_heimdall_tools_are_described_in_human_terms():
    """The point is a researcher glancing at a phone, not a tool name."""
    assert label_for(f"{MCP_PREFIX}mcp_query") == "запрашиваю данные"
    assert label_for(f"{MCP_PREFIX}describe_model") == "изучаю структуру витрины"


def test_an_unknown_tool_degrades_to_its_own_name():
    """A tool added to the surface without a label here must still show up.
    Silently dropping it would make the status claim fewer steps than happened.
    """
    assert label_for(f"{MCP_PREFIX}brand_new") == "brand_new"
    assert label_for("") == "работаю"


# ------------------------------------------------------------- event folding


def test_tool_calls_are_counted_and_heimdall_separately():
    turn = TurnProgress("ses_1", "вопрос")
    turn.observe(_tool_use("ToolSearch"))
    turn.observe(_tool_use(f"{MCP_PREFIX}mcp_query"))
    turn.observe(_tool_use("Bash"))
    snap = turn.snapshot()
    assert snap["tool_calls"] == 3
    assert snap["heimdall_calls"] == 1


def test_repeated_identical_steps_collapse():
    """Paging a mart is five mcp_query calls in a row. Five identical lines
    read as a stutter rather than as progress — but the count must still rise,
    or the status would understate the work."""
    turn = TurnProgress("ses_1", "вопрос")
    for _ in range(5):
        turn.observe(_tool_use(f"{MCP_PREFIX}mcp_query"))
    snap = turn.snapshot()
    assert snap["steps"] == ["запрашиваю данные"]
    assert snap["tool_calls"] == 5


def test_the_rendered_trail_is_bounded():
    """A dozen calls must not produce a status message that grows without end."""
    turn = TurnProgress("ses_1", "вопрос")
    for name in ("get_docs", "list_models", "describe_model", "mcp_query"):
        turn.observe(_tool_use(f"{MCP_PREFIX}{name}"))
    assert turn.snapshot()["status"].count("→") == TRAIL - 1


def test_a_turn_with_no_tools_yet_still_says_something():
    """The window between accepting the question and the first tool call is
    seconds of CLI startup, and it is exactly when someone is staring at it."""
    status = TurnProgress("ses_1", "вопрос").snapshot()["status"]
    assert "думаю" in status


def test_the_result_event_ends_the_turn():
    turn = TurnProgress("ses_1", "вопрос")
    turn.observe(_tool_use(f"{MCP_PREFIX}mcp_query"))
    assert turn.snapshot()["done"] is False
    turn.observe(RESULT)
    snap = turn.snapshot()
    assert snap["done"] is True and snap["failed"] is False
    assert "готово" in snap["status"]


def test_an_errored_result_is_marked_failed():
    turn = TurnProgress("ses_1", "вопрос")
    turn.observe({"type": "result", "is_error": True})
    assert turn.snapshot()["failed"] is True


def test_finish_closes_a_turn_that_never_produced_a_result():
    """A crashed turn emits no `result` event. Without this the poller waits on
    a "думаю…" that will never advance."""
    turn = TurnProgress("ses_1", "вопрос")
    turn.finish(failed=True)
    snap = turn.snapshot()
    assert snap["done"] is True and snap["failed"] is True


def test_unrelated_events_are_ignored():
    turn = TurnProgress("ses_1", "вопрос")
    for event in ({"type": "system", "subtype": "init"},
                  {"type": "user", "message": {"content": []}},
                  {"type": "rate_limit_event"},
                  {}):
        turn.observe(event)
    assert turn.snapshot()["tool_calls"] == 0


# ------------------------------------------------------------------- board


def test_the_board_isolates_sessions():
    board = ProgressBoard()
    a = board.start("ses_a", "q")
    board.start("ses_b", "q")
    a.observe(_tool_use(f"{MCP_PREFIX}mcp_query"))
    assert board.get("ses_a")["tool_calls"] == 1
    assert board.get("ses_b")["tool_calls"] == 0


def test_an_unknown_session_is_absent_rather_than_empty():
    """"Nothing is running" and "something is running silently" must not look
    alike to a poller."""
    assert ProgressBoard().get("ses_missing") is None


def test_a_finished_turn_is_readable_for_a_while():
    """A poller that asks one last time should see "done", not a 404 it would
    read as a session that never ran."""
    board = ProgressBoard(retain_seconds=60)
    board.start("ses_a", "q").observe(RESULT)
    assert board.get("ses_a")["done"] is True


def test_finished_turns_are_eventually_evicted():
    board = ProgressBoard(retain_seconds=0.0)
    turn = board.start("ses_old", "q")
    turn.observe(RESULT)
    time.sleep(0.01)
    board.start("ses_new", "q")          # any start triggers the sweep
    assert board.get("ses_old") is None
    assert board.get("ses_new") is not None


def test_an_in_flight_turn_is_never_evicted():
    """Eviction keyed on `done`, not on age: a slow turn is the one case where
    the progress view matters most."""
    board = ProgressBoard(retain_seconds=0.0)
    board.start("ses_slow", "q")
    board.start("ses_other", "q")
    assert board.get("ses_slow") is not None


def test_a_second_question_replaces_the_view():
    """One entry per session — a session runs one turn at a time."""
    board = ProgressBoard()
    board.start("ses_a", "первый").observe(_tool_use("Bash"))
    board.start("ses_a", "второй")
    assert board.get("ses_a")["tool_calls"] == 0
