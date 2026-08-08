"""LLM spans reconstructed from the CLI's own session transcript.

The model call happens in a subprocess and cannot be instrumented from here.
But Claude Code writes a JSONL transcript of the session, and every assistant
entry in it carries `message.usage` — the per-call token split the root span can
only report as a turn total. Grouped by `message.id`, those entries are the API
calls, one span each.

Verified against a real transcript before this was written (session
ses_4bfcc8f2…, 2026-08-05): 43 assistant rows, 15 distinct message ids, usage on
every one, and no duration field anywhere — hence the timing here is derived
from row timestamps and labelled as derived.
"""
from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sim import telemetry
from sim.agent.claude_code import (
    TranscriptFormatError,
    find_transcript,
    parse_transcript,
)
from sim.fingerprint import RunFingerprint

SESSION = "4936d9a0-44d8-4aae-b963-760a10172dab"


def _root(fingerprint: RunFingerprint):
    return telemetry.start_run(
        "b2e.turn", fingerprint=fingerprint, session_id="ses_test",
        employee_id="1599763", question="сколько витрин?")


def _by_name(exporter: InMemorySpanExporter, name: str) -> list:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _assistant(message_id: str, ts: str, *, out_tokens: int = 252,
               cache_read: int = 12211, cache_write: int = 2566,
               stop: str = "tool_use") -> dict:
    return {
        "type": "assistant", "timestamp": ts, "requestId": "req_x",
        "message": {"id": message_id, "model": "claude-haiku-4-5-20251001",
                    "role": "assistant", "stop_reason": stop,
                    "usage": {"input_tokens": 10, "output_tokens": out_tokens,
                              "cache_read_input_tokens": cache_read,
                              "cache_creation_input_tokens": cache_write},
                    "content": [{"type": "text", "text": "…"}]},
    }


def _user(ts: str) -> dict:
    return {"type": "user", "timestamp": ts,
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}}


def _write(tmp_path, rows, name: str = SESSION):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    return path


# --------------------------------------------------------------- locating


def test_the_transcript_is_found_by_session_id_not_by_rebuilding_the_path(tmp_path):
    """The CLI derives the directory name from the working directory by rules
    this code does not own. Searching for the session's own file survives them
    changing; reimplementing the escaping does not."""
    project = tmp_path / ".claude" / "projects" / "-app-var-sessions-ses-abc"
    project.mkdir(parents=True)
    wanted = _write(project, [_assistant("msg_1", "2026-08-05T15:35:29.244Z")])

    assert find_transcript(str(tmp_path), SESSION) == wanted


def test_a_missing_transcript_is_not_an_error(tmp_path):
    """An older CLI, or a turn that died before writing one. The turn still
    produced an answer and must not be failed over its telemetry."""
    assert find_transcript(str(tmp_path), SESSION) is None


# ---------------------------------------------------------------- parsing


def test_rows_sharing_one_message_id_are_one_api_call(tmp_path):
    """A single response arrives as several rows — text block, tool_use block.
    Counting rows would have reported 43 model calls for a turn that made 15."""
    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z"),
        _assistant("msg_1", "2026-08-05T15:35:30.006Z"),
        _assistant("msg_2", "2026-08-05T15:35:33.856Z"),
    ])
    calls = parse_transcript(path, since=0.0)
    assert [c.message_id for c in calls] == ["msg_1", "msg_2"]


def test_the_call_carries_the_token_split_the_root_span_can_only_total(tmp_path):
    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:33.856Z", out_tokens=252,
                   cache_read=12211, cache_write=2566)])
    call = parse_transcript(path, since=0.0)[0]
    assert (call.input_tokens, call.output_tokens) == (10, 252)
    assert (call.cache_read_tokens, call.cache_creation_tokens) == (12211, 2566)
    assert call.model == "claude-haiku-4-5-20251001"
    assert call.stop_reason == "tool_use"


def test_only_this_turns_calls_are_read(tmp_path):
    """A resumed session appends to one file. Without the cutoff, turn two would
    re-emit turn one's calls and every token count would double."""
    path = _write(tmp_path, [
        _assistant("msg_old", "2026-08-05T15:00:00.000Z"),
        _assistant("msg_new", "2026-08-05T15:35:33.856Z"),
    ])
    cutoff = 1785943600.0                       # 2026-08-05T15:26:40Z
    calls = parse_transcript(path, since=cutoff)
    assert [c.message_id for c in calls] == ["msg_new"]


def test_a_transcript_in_an_unknown_shape_fails_loudly(tmp_path):
    """The format is undocumented and tied to one CLI version. A parser that
    quietly returns nothing would recreate exactly the defect this whole effort
    exists to fix: an absent measurement that reads as a measured zero."""
    path = _write(tmp_path, [{"type": "assistant", "payload": "shape changed"},
                             {"type": "assistant", "payload": "again"}])
    with pytest.raises(TranscriptFormatError):
        parse_transcript(path, since=0.0)


def test_a_transcript_with_no_assistant_rows_at_all_is_not_a_format_error(tmp_path):
    """A turn refused before the first model call writes a transcript with none.
    That is a real state, not a broken parser."""
    path = _write(tmp_path, [_user("2026-08-05T15:35:29.244Z")])
    assert parse_transcript(path, since=0.0) == []


# ----------------------------------------------------------------- spans


def test_each_api_call_becomes_one_llm_span_under_the_turn(spans, fingerprint,
                                                           tmp_path):
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z"),
        _assistant("msg_1", "2026-08-05T15:35:30.006Z"),
        _assistant("msg_2", "2026-08-05T15:35:33.856Z", stop="end_turn"),
    ])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root)

    llm = _by_name(spans, "llm.messages.create")
    assert len(llm) == 2
    root_span = _by_name(spans, "b2e.turn")[0]
    assert {s.parent.span_id for s in llm} == {root_span.context.span_id}


def test_the_llm_span_reports_tokens_where_phoenix_reads_them(spans, fingerprint,
                                                              tmp_path):
    """Unlike the AGENT root, an LLM span does populate Phoenix's own token
    columns — that is the point of putting them on this kind of span."""
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [_assistant("msg_1", "2026-08-05T15:35:29.244Z")])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root)

    attrs = _by_name(spans, "llm.messages.create")[0].attributes
    assert attrs["llm.token_count.prompt"] == 10 + 12211 + 2566
    assert attrs["llm.token_count.completion"] == 252
    assert attrs["llm.token_count.prompt_details.cache_read"] == 12211
    assert attrs["llm.model_name"] == "claude-haiku-4-5-20251001"
    assert attrs["llm.finish_reason"] == "tool_use"


def test_the_span_duration_is_labelled_as_derived_not_measured(spans, fingerprint,
                                                               tmp_path):
    """The transcript has no duration field — checked: no ttft, no latency, no
    elapsed. The window between two real timestamps is the best available, and
    it must not be presented as though something timed the call."""
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [
        _user("2026-08-05T15:35:29.000Z"),
        _assistant("msg_1", "2026-08-05T15:35:31.500Z"),
    ])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root)

    span = _by_name(spans, "llm.messages.create")[0]
    assert span.attributes["b2e.llm.timing"] == "derived"
    assert (span.end_time - span.start_time) / 1e9 == pytest.approx(2.5, abs=0.01)


# ------------------------------------------------------------- the harness


@pytest.fixture()
def harness(tmp_path):
    import pathlib

    from sim.agent.claude_code import ClaudeCodeHarness
    bridge = pathlib.Path(__file__).resolve().parents[1] / "heimdall" / "bridge.py"
    return ClaudeCodeHarness(
        heimdall_url="http://127.0.0.1:1", heimdall_token="t",
        bridge_path=str(bridge), runner_path="/opt/skills/run",
        session_root=str(tmp_path / "sessions"),
        claude_home=str(tmp_path / "home"))


def test_a_stateless_turn_now_leaves_a_transcript_to_read(harness):
    """Verified on the live stack before this changed: persistence files a
    transcript, it does not share context — a second turn in the same directory
    with no --resume had no memory of the first. Independence comes from not
    resuming, which stateless turns still never do."""
    from sim.agent.config import AgentConfig

    argv = harness.build_argv("вопрос", config=AgentConfig(),
                              system_suffix="s", mcp_config_path="/tmp/mcp.json")
    assert "--no-session-persistence" not in argv
    assert "--resume" not in argv


def test_a_stateless_turn_still_never_resumes(harness):
    from sim.agent.config import AgentConfig

    argv = harness.build_argv("вопрос", config=AgentConfig(),
                              system_suffix="s", mcp_config_path="/tmp/mcp.json",
                              resume_session_id="should-be-ignored")
    assert "--resume" not in argv


def test_an_unreadable_transcript_is_visible_on_the_turn(spans, fingerprint):
    """The rule this stack now runs on: an instrumentation gap must show up in
    the data, not only in a log nobody reads."""
    from sim.agent.claude_code import mark_transcript_gap

    with _root(fingerprint) as root:
        mark_transcript_gap(root, "format")

    assert _by_name(spans, "b2e.turn")[0].attributes["b2e.trace.llm_spans"] == "format"
