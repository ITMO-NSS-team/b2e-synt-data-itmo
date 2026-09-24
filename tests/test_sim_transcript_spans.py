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
               stop: str = "tool_use", blocks: list | None = None) -> dict:
    return {
        "type": "assistant", "timestamp": ts, "requestId": "req_x",
        "message": {"id": message_id, "model": "claude-haiku-4-5-20251001",
                    "role": "assistant", "stop_reason": stop,
                    "usage": {"input_tokens": 10, "output_tokens": out_tokens,
                              "cache_read_input_tokens": cache_read,
                              "cache_creation_input_tokens": cache_write},
                    "content": blocks if blocks is not None
                    else [{"type": "text", "text": "…"}]},
    }


def _thinking(text: str, signature: str = "CAIS8QMKhwEIEBgC") -> dict:
    return {"type": "thinking", "thinking": text, "signature": signature}


def _tool_use(name: str, call_id: str = "toolu_1", **args) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


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


def test_the_llm_span_reports_tokens_where_backends_read_them(
        spans, fingerprint, tmp_path, monkeypatch):
    """Unlike the AGENT root, an LLM span does populate Phoenix's own token
    columns — that is the point of putting them on this kind of span."""
    from sim.agent.claude_code import emit_llm_spans

    monkeypatch.setenv("LLM_PROVIDER", "zai")
    path = _write(tmp_path, [_assistant("msg_1", "2026-08-05T15:35:29.244Z")])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root)

    attrs = _by_name(spans, "llm.messages.create")[0].attributes
    assert attrs["llm.token_count.prompt"] == 10 + 12211 + 2566
    assert attrs["llm.token_count.completion"] == 252
    assert attrs["llm.token_count.prompt_details.cache_read"] == 12211
    assert attrs["llm.model_name"] == "claude-haiku-4-5-20251001"
    assert attrs["llm.finish_reason"] == "tool_use"
    assert attrs["llm.provider"] == "zai"
    assert attrs["b2e.llm.protocol"] == "anthropic"
    assert attrs["gen_ai.provider.name"] == "zai"
    assert attrs["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert attrs["gen_ai.usage.input_tokens"] == 10 + 12211 + 2566
    assert attrs["gen_ai.usage.output_tokens"] == 252
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 12211
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 2566


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


# -------------------------------------------------------- reasoning content


def test_the_reasoning_is_recovered_and_named_as_the_convention_names_it(tmp_path):
    """The thing this whole layer exists for.

    Headless sessions persist reasoning in full — measured on the deployed
    stack: 291 thinking blocks across 16 sessions, every one non-empty — and
    until 2026-08-09 the parser read the token counts off these rows and threw
    the text away. `thinking` becomes `reasoning` here because that is the word
    the installed semconv uses; a provider synonym would hide the field from
    every other reader of the span store.
    """
    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z",
                   blocks=[_thinking("Сначала загружу инструменты Heimdall.")]),
    ])
    call = parse_transcript(path, since=0.0)[0]
    assert [c["type"] for c in call.contents] == ["reasoning"]
    assert call.reasoning == "Сначала загружу инструменты Heimdall."
    assert call.contents[0]["signature"] == "CAIS8QMKhwEIEBgC"


def test_the_later_rows_of_one_message_are_merged_not_dropped(tmp_path):
    """One response is written one row per content block. Keeping only the
    first — which is what `if message_id in calls: continue` did — kept the
    reasoning and discarded the answer, or the reverse, depending on order."""
    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z",
                   blocks=[_thinking("Надо посчитать.")]),
        _assistant("msg_1", "2026-08-05T15:35:29.900Z",
                   blocks=[{"type": "text", "text": "Считаю."}]),
        _assistant("msg_1", "2026-08-05T15:35:30.500Z",
                   blocks=[_tool_use("mcp__heimdall__mcp_query", schema="hr")]),
    ])
    calls = parse_transcript(path, since=0.0)
    assert len(calls) == 1
    assert [c["type"] for c in calls[0].contents] == ["reasoning", "text"]
    assert [t["name"] for t in calls[0].tool_calls] == ["mcp__heimdall__mcp_query"]
    assert json.loads(calls[0].tool_calls[0]["arguments"]) == {"schema": "hr"}


def test_the_window_closes_at_the_last_row_of_the_message(tmp_path):
    """A model that reasons first writes its reasoning row before its answer
    row. Closing the span at the first row ends the call before the answer
    existed."""
    path = _write(tmp_path, [
        _user("2026-08-05T15:35:29.000Z"),
        _assistant("msg_1", "2026-08-05T15:35:30.000Z",
                   blocks=[_thinking("…")]),
        _assistant("msg_1", "2026-08-05T15:35:32.000Z",
                   blocks=[{"type": "text", "text": "ответ"}]),
    ])
    call = parse_transcript(path, since=0.0)[0]
    assert (call.ended_ns - call.started_ns) / 1e9 == pytest.approx(3.0, abs=0.01)


def test_a_redacted_thinking_block_stays_a_reasoning_block(tmp_path):
    """It carries no text at all. The convention has a field for exactly this
    and names Anthropic in its docstring, so it must not become an absence —
    a redacted thought is not the same as no thought."""
    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z",
                   blocks=[{"type": "redacted_thinking", "data": "AAAA"}]),
    ])
    call = parse_transcript(path, since=0.0)[0]
    assert call.contents == [{"type": "reasoning", "data": "AAAA"}]


def test_the_reasoning_lands_where_phoenix_re_nests_it(spans, fingerprint, tmp_path):
    """Indexed flat keys, not a JSON blob. Verified against the deployed Phoenix
    19.13.0 on 2026-08-09: the flat form is re-nested into
    `llm.output_messages[0].message.contents[…]`, which the UI renders; a JSON
    string is stored as a string and stays opaque."""
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z",
                   blocks=[_thinking("рассуждение"),
                           {"type": "text", "text": "ответ"},
                           _tool_use("ToolSearch", query="select:x")]),
    ])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root)

    a = _by_name(spans, "llm.messages.create")[0].attributes
    out = "llm.output_messages.0.message"
    assert a[f"{out}.role"] == "assistant"
    assert a[f"{out}.contents.0.message_content.type"] == "reasoning"
    assert a[f"{out}.contents.0.message_content.text"] == "рассуждение"
    assert a[f"{out}.contents.0.message_content.signature"] == "CAIS8QMKhwEIEBgC"
    assert a[f"{out}.contents.1.message_content.type"] == "text"
    assert a[f"{out}.tool_calls.0.tool_call.function.name"] == "ToolSearch"
    assert a[f"{out}.tool_calls.0.tool_call.id"] == "toolu_1"


def test_the_visible_answer_excludes_the_reasoning(spans, fingerprint, tmp_path):
    """`output.value` is what the scorer scans for fabricated identifiers. A
    number the model considered and rejected in its reasoning must not be found
    there as though it had been said."""
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z", stop="end_turn",
                   blocks=[_thinking("Может быть 294000, но проверю."),
                           {"type": "text", "text": "В компании 2741 сотрудник."}]),
    ])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root)

    a = _by_name(spans, "llm.messages.create")[0].attributes
    assert a["output.value"] == "В компании 2741 сотрудник."
    assert "294000" not in a["output.value"]


# ------------------------------------------------- the reconstructed prompt


def test_the_prompt_is_the_conversation_before_the_call_not_after(spans, fingerprint,
                                                                  tmp_path):
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [
        {"type": "user", "timestamp": "2026-08-05T15:35:29.000Z",
         "message": {"role": "user", "content": "сколько витрин?"}},
        _assistant("msg_1", "2026-08-05T15:35:30.000Z",
                   blocks=[_tool_use("mcp__heimdall__list_models")]),
        _user("2026-08-05T15:35:31.000Z"),
        _assistant("msg_2", "2026-08-05T15:35:32.000Z", stop="end_turn",
                   blocks=[{"type": "text", "text": "три"}]),
    ])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root)

    first, second = _by_name(spans, "llm.messages.create")
    # The first call saw only the question.
    assert first.attributes["llm.input_messages.0.message.role"] == "user"
    assert first.attributes["llm.input_messages.0.message.content"] == "сколько витрин?"
    assert "llm.input_messages.1.message.role" not in first.attributes
    # The second saw the question, its own prior response, and the tool result.
    assert second.attributes["llm.input_messages.1.message.role"] == "assistant"
    assert second.attributes["llm.input_messages.2.message.role"] == "tool"
    assert second.attributes["llm.input_messages.2.message.tool_call_id"] == "t1"


def test_the_prompt_says_it_is_a_reconstruction(spans, fingerprint, tmp_path):
    """Measured on a real turn: the first call reported ~12 200 prompt tokens
    for a 61-character question, and the harness can account for perhaps a
    thousand of them. Anyone reading these messages as *the* prompt has to meet
    the flag before they reach a conclusion."""
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [_assistant("msg_1", "2026-08-05T15:35:29.244Z")])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root,
                       system="Ты — корпоративный ассистент.")

    a = _by_name(spans, "llm.messages.create")[0].attributes
    assert a["b2e.llm.prompt_reconstruction"] == "conversation_only"
    assert a["b2e.llm.prompt_missing"] == "cli_system_prompt,tool_schemas"
    assert a["llm.system"] == "Ты — корпоративный ассистент."
    assert a["b2e.llm.system_partial"] is True


def test_a_tool_result_in_the_prompt_is_capped_and_says_where_the_rest_is(tmp_path):
    """The full text is on the tool span. Repeating it in every later call's
    prompt is quadratic — on the longest session measured, 49 calls over 713 KB
    of tool output, the same payloads would be written some 1 200 times."""
    from sim.agent.claude_code import MAX_PROMPT_TOOL_RESULT_CHARS, TRUNCATION_NOTE

    big = "x" * (MAX_PROMPT_TOOL_RESULT_CHARS + 500)
    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.000Z"),
        {"type": "user", "timestamp": "2026-08-05T15:35:30.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "t1", "content": big}]}},
        _assistant("msg_2", "2026-08-05T15:35:31.000Z"),
    ])
    call = parse_transcript(path, since=0.0)[1]
    result = call.input_messages[-1]["content"]
    assert result.endswith(TRUNCATION_NOTE)
    assert len(result) < len(big)


def test_an_elided_prompt_prefix_is_counted_not_silently_shortened(spans, fingerprint,
                                                                   tmp_path):
    from sim.agent.claude_code import MAX_PROMPT_MESSAGES, emit_llm_spans

    total = MAX_PROMPT_MESSAGES + 5
    rows = [_assistant(f"msg_{i}", f"2026-08-05T15:{35 + i // 60}:"
                                   f"{i % 60:02d}.000Z") for i in range(total)]
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(_write(tmp_path, rows), since=0.0), root=root)

    last = _by_name(spans, "llm.messages.create")[-1].attributes
    # The final call's prefix is every message but its own, hence total - 1.
    assert last["b2e.llm.input_messages_elided"] == (total - 1) - MAX_PROMPT_MESSAGES
    assert f"llm.input_messages.{MAX_PROMPT_MESSAGES}.message.role" not in last
    # The question is kept even when the middle goes: it is what the turn is about.
    assert last["llm.input_messages.0.message.role"] == "assistant"


def test_history_before_the_cutoff_is_context_even_though_it_is_not_a_call(tmp_path):
    """A resumed session appends to one file. Turn one's messages make no call
    of their own — that would double every token count — but they *are* what
    turn two's calls were sent."""
    path = _write(tmp_path, [
        _assistant("msg_old", "2026-08-05T15:00:00.000Z",
                   blocks=[{"type": "text", "text": "прошлый ответ"}]),
        _assistant("msg_new", "2026-08-05T15:35:33.856Z"),
    ])
    calls = parse_transcript(path, since=1785943600.0)
    assert [c.message_id for c in calls] == ["msg_new"]
    assert calls[0].input_messages[0]["contents"][0]["text"] == "прошлый ответ"


def test_content_capture_can_be_switched_off_visibly(spans, fingerprint, tmp_path):
    """Off must not look like a model that did not reason. That is the same
    failure the 0.000 s tool spans were: an absent measurement reading as a
    measured zero."""
    from sim.agent.claude_code import emit_llm_spans

    path = _write(tmp_path, [
        _assistant("msg_1", "2026-08-05T15:35:29.244Z", blocks=[_thinking("…")])])
    with _root(fingerprint) as root:
        emit_llm_spans(parse_transcript(path, since=0.0), root=root, content=False)

    a = _by_name(spans, "llm.messages.create")[0].attributes
    assert a["b2e.trace.llm_content"] == "disabled"
    assert not [k for k in a if k.startswith("llm.output_messages")]
    assert a["llm.token_count.completion"] == 252      # the counts still stand


# -------------------------------------------- iterations and measured ttft


def _stream_assistant(message_id: str, blocks: list) -> dict:
    return {"type": "assistant",
            "message": {"id": message_id, "role": "assistant", "content": blocks}}


def _stream_result(message_id: str, ttft_ms: int) -> dict:
    return {"type": "stream_event", "ttft_ms": ttft_ms,
            "event": {"type": "message_start",
                      "message": {"id": message_id,
                                  "model": "claude-haiku-4-5-20251001"}}}


def test_the_tree_finally_has_an_iteration_level(spans, fingerprint):
    """`docs/span-schema.md` called this level unobservable outside the CLI.
    It is observable: the stream names the message, and one message id is one
    iteration."""
    from sim.agent.claude_code import ToolSpanRecorder

    with _root(fingerprint) as root:
        rec = ToolSpanRecorder(root)
        rec.observe(_stream_assistant("msg_1", [
            {"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]))
        rec.observe({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}})
        rec.observe(_stream_assistant("msg_2", [{"type": "text", "text": "готово"}]))
        rec.finish()

    iterations = [s for s in spans.get_finished_spans()
                  if s.name.startswith("iteration.")]
    assert [s.name for s in iterations] == ["iteration.1", "iteration.2"]
    assert iterations[0].attributes["b2e.llm.message_id"] == "msg_1"
    assert iterations[0].attributes["openinference.span.kind"] == "CHAIN"


def test_a_tool_call_is_filed_under_the_iteration_that_asked_for_it(spans,
                                                                    fingerprint):
    from sim.agent.claude_code import ToolSpanRecorder

    with _root(fingerprint) as root:
        rec = ToolSpanRecorder(root)
        rec.observe(_stream_assistant("msg_1", [
            {"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]))
        rec.observe({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}})
        rec.finish()

    tool = _by_name(spans, "tool.ToolSearch")[0]
    iteration = [s for s in spans.get_finished_spans()
                 if s.name == "iteration.1"][0]
    assert tool.parent.span_id == iteration.context.span_id


def test_the_model_call_lands_inside_its_own_iteration(spans, fingerprint,
                                                       tmp_path):
    """Matched by message id, which the stream and the transcript both name —
    exact, not a time heuristic."""
    from sim.agent.claude_code import ToolSpanRecorder, emit_llm_spans

    path = _write(tmp_path, [_assistant("msg_1", "2026-08-05T15:35:29.244Z")])
    with _root(fingerprint) as root:
        rec = ToolSpanRecorder(root)
        rec.observe(_stream_assistant("msg_1", [{"type": "text", "text": "…"}]))
        rec.finish()
        emit_llm_spans(parse_transcript(path, since=0.0), root=root, recorder=rec)

    llm = _by_name(spans, "llm.messages.create")[0]
    iteration = [s for s in spans.get_finished_spans()
                 if s.name == "iteration.1"][0]
    assert llm.parent.span_id == iteration.context.span_id


def test_time_to_first_token_is_measured_and_says_so(spans, fingerprint, tmp_path):
    """The transcript has no duration of any kind, so the span window stays
    derived. `--include-partial-messages` reports a real ttft on
    `message_start`, and the two must not be confused for one another."""
    from sim.agent.claude_code import ToolSpanRecorder, emit_llm_spans

    path = _write(tmp_path, [_assistant("msg_1", "2026-08-05T15:35:29.244Z")])
    with _root(fingerprint) as root:
        rec = ToolSpanRecorder(root)
        rec.observe(_stream_result("msg_1", 1811))
        rec.observe(_stream_assistant("msg_1", [{"type": "text", "text": "…"}]))
        rec.finish()
        emit_llm_spans(parse_transcript(path, since=0.0), root=root, recorder=rec)

    a = _by_name(spans, "llm.messages.create")[0].attributes
    assert a["b2e.llm.ttft_ms"] == 1811
    assert a["b2e.llm.ttft_source"] == "measured"
    assert a["b2e.llm.timing"] == "derived"          # the window still is


def test_the_partial_message_flag_is_passed_because_nothing_else_times_a_call(
        harness):
    from sim.agent.config import AgentConfig

    argv = harness.build_argv("вопрос", config=AgentConfig(),
                              system_suffix="s", mcp_config_path="/tmp/mcp.json")
    assert "--include-partial-messages" in argv


def test_the_turns_own_measured_timings_reach_the_root(spans, fingerprint):
    """`duration_api_ms` is the CLI's figure and can exceed the turn's wall
    time, so it is recorded and not subtracted from anything."""
    from sim.agent.claude_code import emit_spans, parse_stream

    stream = json.dumps({
        "type": "result", "result": "готово", "session_id": "s", "num_turns": 1,
        "duration_ms": 3898, "duration_api_ms": 5732, "ttft_ms": 3458,
        "ttft_stream_ms": 2392, "time_to_request_ms": 56,
        "usage": {"input_tokens": 10, "output_tokens": 5}})
    with _root(fingerprint) as root:
        emit_spans(parse_stream(stream), root=root)

    a = _by_name(spans, "b2e.turn")[0].attributes
    assert a["b2e.turn.api_duration_ms"] == 5732
    assert a["b2e.turn.ttft_ms"] == 3458
    assert a["b2e.turn.ttft_stream_ms"] == 2392
    assert a["b2e.turn.time_to_request_ms"] == 56


def test_an_unreadable_transcript_is_visible_on_the_turn(spans, fingerprint):
    """The rule this stack now runs on: an instrumentation gap must show up in
    the data, not only in a log nobody reads."""
    from sim.agent.claude_code import mark_transcript_gap

    with _root(fingerprint) as root:
        mark_transcript_gap(root, "format")

    assert _by_name(spans, "b2e.turn")[0].attributes["b2e.trace.llm_spans"] == "format"
