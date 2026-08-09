"""The trace explorer: reshaping, analysis, and the one CSP exception.

Two sources feed one viewer — Postgres for the offline export, Phoenix's REST
API for the live tab — and they do not agree on span shape. Everything here is
about the seam between them, plus the security question the live page raises:
it renders agent-authored text under a policy that permits inline script.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sim import traceview
from sim.admin.app import AdminState, create_app
from sim.admin.security import headers_for

AUTH = ("admin", "test-password")


# ------------------------------------------------------------- reshaping


def test_dotted_attributes_become_nested_structures():
    """Postgres stores attributes nested; the REST API flattens them. One of
    the two has to be converted or the viewer reads neither."""
    out = traceview.unflatten({
        "b2e.run.model_id": "claude-haiku-4-5-20251001",
        "b2e.turn.cost_usd": 0.09,
        "llm.token_count.prompt": 12226,
    })
    assert out["b2e"]["run"]["model_id"] == "claude-haiku-4-5-20251001"
    assert out["b2e"]["turn"]["cost_usd"] == 0.09
    assert out["llm"]["token_count"]["prompt"] == 12226


def test_indexed_attributes_become_lists_not_objects_keyed_by_digits():
    """The message layout is an indexed sequence. A viewer handed
    {"0": …, "1": …} where it expects an array renders nothing and says
    nothing, which is the failure mode worth a test of its own."""
    out = traceview.unflatten({
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.contents.0.message_content.type": "reasoning",
        "llm.output_messages.0.message.contents.0.message_content.text": "думаю",
        "llm.output_messages.0.message.contents.1.message_content.type": "text",
        "llm.output_messages.0.message.contents.1.message_content.text": "ответ",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "ToolSearch",
    })
    messages = out["llm"]["output_messages"]
    assert isinstance(messages, list) and len(messages) == 1
    contents = messages[0]["message"]["contents"]
    assert isinstance(contents, list)
    assert [c["message_content"]["type"] for c in contents] == ["reasoning", "text"]
    calls = messages[0]["message"]["tool_calls"]
    assert calls[0]["tool_call"]["function"]["name"] == "ToolSearch"


def test_indices_beyond_nine_keep_their_numeric_order():
    """Sorted as strings, message 10 lands between 1 and 2 and the whole
    conversation is silently reordered."""
    flat = {f"llm.input_messages.{i}.message.role": str(i) for i in range(12)}
    roles = traceview.unflatten(flat)["llm"]["input_messages"]
    assert [m["message"]["role"] for m in roles] == [str(i) for i in range(12)]


def test_a_node_with_a_named_sibling_stays_an_object():
    """Only an all-integer node becomes a list. Otherwise an attribute that
    happens to be called `0` would reorder its neighbours out of existence."""
    out = traceview.unflatten({"a.0": "x", "a.name": "y"})
    assert out["a"] == {"0": "x", "name": "y"}


def test_a_rest_span_is_reshaped_into_what_the_viewer_reads():
    span = traceview.normalise_rest_span({
        "context": {"span_id": "abc123", "trace_id": "t1"},
        "parent_id": "parent1", "name": "llm.messages.create", "span_kind": "LLM",
        "start_time": "2026-08-09T05:16:48.000000+00:00",
        "end_time": "2026-08-09T05:16:50.500000+00:00",
        "attributes": {"b2e.llm.ttft_ms": 1386},
    })
    assert span["span_id"] == "abc123"
    assert span["duration_ms"] == pytest.approx(2500, abs=1)
    assert span["attributes"]["b2e"]["llm"]["ttft_ms"] == 1386


# -------------------------------------------------------------- analysis


def _span(kind, name, span_id, parent, attrs=None):
    return {"span_id": span_id, "parent_id": parent, "name": name,
            "span_kind": kind, "duration_ms": 10.0, "attributes": attrs or {}}


def _turn(trace_id="t1", *, reasoning="почему"):
    root = _span("AGENT", "b2e.turn", "r", None, {
        "b2e": {"harness": "claude_code", "turn": {"cost_usd": 0.1},
                "run": {"model_id": "claude-haiku-4-5-20251001"}},
        "llm": {"token_count": {"prompt": 100, "completion": 10}},
        "input": {"value": "вопрос"}, "output": {"value": "ответ"},
        "session": {"id": "ses_x"}, "user": {"id": "1599763"}})
    iteration = _span("CHAIN", "iteration.1", "i1", "r")
    llm = _span("LLM", "llm.messages.create", "l1", "i1", {
        "llm": {"token_count": {"prompt": 100, "completion": 10},
                "output_messages": [{"message": {"role": "assistant", "contents": [
                    {"message_content": {"type": "reasoning", "text": reasoning}}]}}]},
        "b2e": {"llm": {"ttft_ms": 1386}}})
    tool = _span("TOOL", "tool.ToolSearch", "s1", "i1")
    return {"trace_id": trace_id, "started_utc": "2026-08-09T05:16:48Z",
            "duration_s": 1.0, "spans": [root, iteration, llm, tool]}


def test_a_turn_is_summarised_from_its_spans():
    trace = traceview.enrich(_turn())
    assert trace["layer_counts"] == {"iteration_CHAIN": 1, "LLM": 1, "TOOL": 1,
                                     "heimdall_CHAIN": 0, "sandbox_CHAIN": 0}
    assert trace["reasoning"]["spans_with_reasoning"] == 1
    assert trace["reasoning"]["ttft_ms"]["median"] == 1386
    assert trace["user_prompt"] == "вопрос"
    assert trace["run_fingerprint"]["model_id"] == "claude-haiku-4-5-20251001"


def test_a_clean_corpus_reports_verifications_not_anomalies():
    traces = [traceview.enrich(_turn("t1")), traceview.enrich(_turn("t2"))]
    document = traceview.build_document(traces, {})
    kinds = {f["id"]: f["kind"] for f in document["_findings"]}
    assert kinds["iteration-level-exists"] == "verification"
    assert kinds["tree-is-fully-parented"] == "verification"
    assert kinds["llm-spans-reconcile-exactly"] == "verification"
    assert kinds["reasoning-is-present"] == "verification"


def test_a_pre_change_trace_is_reported_as_an_anomaly_not_hidden():
    """Traces recorded before 2026-08-09 have no iteration layer. The viewer
    must say so rather than render them as though they matched."""
    old = _turn("old")
    for span in old["spans"]:
        if span["name"] == "iteration.1":
            old["spans"].remove(span)
            break
    for span in old["spans"]:
        if span["span_kind"] in ("LLM", "TOOL"):
            span["parent_id"] = "r"
    document = traceview.build_document([traceview.enrich(old)], {})
    kinds = {f["id"]: f["kind"] for f in document["_findings"]}
    assert kinds["tree-is-fully-parented"] == "anomaly"


def test_a_permitted_file_reading_bash_call_is_reported_as_a_defect():
    """Found in real traces on 2026-08-09: `grep` is not refused by the
    matcher, so the agent read its own session transcript despite Read, Glob
    and Grep all being denied."""
    trace = _turn()
    trace["spans"].append(_span("TOOL", "tool.Bash", "b1", "i1", {
        "input": {"value": json.dumps({"command": "grep -o 'x' /app/var/t.jsonl"})},
        "output": {"value": "some file content"}}))
    document = traceview.build_document([traceview.enrich(trace)], {})
    defect = [f for f in document["_findings"] if f["kind"] == "defect"]
    assert defect and defect[0]["id"] == "bash-file-read-bypasses-the-tool-policy"
    assert document["_findings_detail"]["bash_file_reads"][0]["denied"] is False


@pytest.mark.parametrize("refusal", [
    "Permission to use Bash has been denied because Claude Code is running in …",
    "Claude requested permissions to use Bash, but you haven't granted it yet",
])
def test_a_refused_bash_call_is_not_counted_as_a_bypass(refusal):
    """Both wordings, because the harness changed permission mode and a corpus
    spans both. Matching only the current one scores every older refusal as a
    successful read."""
    trace = _turn()
    trace["spans"].append(_span("TOOL", "tool.Bash", "b1", "i1", {
        "input": {"value": json.dumps({"command": "cat /etc/passwd"})},
        "output": {"value": refusal}}))
    document = traceview.build_document([traceview.enrich(trace)], {})
    assert not [f for f in document["_findings"] if f["kind"] == "defect"]
    assert document["_findings_detail"]["bash_file_reads"][0]["outcome"] == "denied"


def test_a_bash_call_with_no_recorded_output_is_not_scored_either_way():
    """Tool results were not captured until 2026-08-08. An empty output means
    the trace does not say, and calling that a successful read would
    manufacture a finding out of an instrumentation gap."""
    trace = _turn()
    trace["spans"].append(_span("TOOL", "tool.Bash", "b1", "i1", {
        "input": {"value": json.dumps({"command": "head -c 500 /app/var/t.jsonl"})}}))
    document = traceview.build_document([traceview.enrich(trace)], {})
    assert not [f for f in document["_findings"] if f["kind"] == "defect"]
    assert document["_findings_detail"]["bash_file_reads"][0]["outcome"] == "unrecorded"


def test_unscoreable_calls_are_counted_beside_the_confirmed_ones():
    trace = _turn()
    trace["spans"].append(_span("TOOL", "tool.Bash", "b1", "i1", {
        "input": {"value": json.dumps({"command": "grep x /app/var/t.jsonl"})},
        "output": {"value": "content"}}))
    trace["spans"].append(_span("TOOL", "tool.Bash", "b2", "i1", {
        "input": {"value": json.dumps({"command": "cat /app/var/t.jsonl"})}}))
    document = traceview.build_document([traceview.enrich(trace)], {})
    defect = [f for f in document["_findings"] if f["kind"] == "defect"][0]
    assert "1 Bash calls read a file" in defect["statement"]
    assert "A further 1 such calls have no recorded output" in defect["statement"]


def test_the_live_view_drops_reconstructed_prompts_and_says_how_many():
    """They are quadratic in a turn's length and show nothing the individual
    spans do not. Dropping them silently would be the same defect this whole
    schema argues against."""
    trace = _turn()
    trace["spans"][2]["attributes"]["llm"]["input_messages"] = [
        {"message": {"role": "user", "content": "a"}},
        {"message": {"role": "tool", "content": "b"}}]
    dropped = traceview.strip_input_messages([trace])
    assert dropped == 2
    llm = trace["spans"][2]["attributes"]["llm"]
    assert "input_messages" not in llm
    assert llm["input_messages_dropped_from_view"] == 2


# ------------------------------------------------------------- rendering


def test_the_payload_cannot_close_the_script_tag_that_carries_it():
    """The security property the explorer's CSP exception rests on. The data is
    agent-authored: reasoning, tool results, whatever Heimdall returned. Under
    script-src 'unsafe-inline' a payload that escaped its container would be
    executable."""
    trace = _turn(reasoning="</script><script>alert(1)</script>")
    page = traceview.render(traceview.build_document([traceview.enrich(trace)], {}),
                            template="<html>__DATA__</html>")
    assert "</script>" not in page
    assert "<\\/script>" in page
    # …and the escape is a legal JSON escape, so nothing about the data changed.
    payload = page[len("<html>"):-len("</html>")]
    restored = json.loads(payload.replace("<\\/", "</"))
    text = (restored["traces"][0]["spans"][2]["attributes"]["llm"]
            ["output_messages"][0]["message"]["contents"][0]["message_content"]["text"])
    assert text == "</script><script>alert(1)</script>"


def test_the_shipped_template_has_a_placeholder_to_fill():
    """The page is also built offline by build_viewer.py from the same file. If
    the placeholder is ever renamed, both callers produce a viewer with no data
    and no error."""
    assert traceview.TEMPLATE.is_file(), traceview.TEMPLATE
    assert traceview.PLACEHOLDER in traceview.TEMPLATE.read_text(encoding="utf-8")


# ------------------------------------------------------ the CSP exception


def test_only_the_explorer_relaxes_the_script_policy():
    assert "script-src" not in headers_for("/")["Content-Security-Policy"]
    assert "script-src" not in headers_for("/skills")["Content-Security-Policy"]
    explorer = headers_for("/explorer")["Content-Security-Policy"]
    assert "script-src 'unsafe-inline'" in explorer


def test_the_exception_is_matched_exactly_not_by_prefix():
    """A prefix rule is how one exception becomes a general one."""
    assert "script-src" not in headers_for("/explorer/evil")["Content-Security-Policy"]
    assert "script-src" not in headers_for("/explorerish")["Content-Security-Policy"]


def test_the_relaxed_policy_still_forbids_every_remote_origin():
    """Inline script and nothing else: the page cannot fetch, beacon, or load a
    script it did not ship with."""
    policy = headers_for("/explorer")["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "connect-src" not in policy
    assert "img-src" not in policy


# ------------------------------------------------------------- the route


class _FakePhoenix:
    """Phoenix as the REST client sees it, in the shape it really returns:
    attributes flattened by dotted key, span id under `context`."""

    def __init__(self, traces=None, error=None):
        self._traces, self._error = traces or [], error

    def latest_traces(self, *, limit=100, span_cap=20000):
        if self._error:
            raise self._error
        return self._traces[:limit]


def _rest_trace(trace_id="t1"):
    return {
        "trace_id": trace_id,
        "start_time": "2026-08-09T05:16:48+00:00",
        "end_time": "2026-08-09T05:17:48+00:00",
        "spans": [
            {"context": {"span_id": "r", "trace_id": trace_id}, "parent_id": None,
             "name": "b2e.turn", "span_kind": "AGENT",
             "start_time": "2026-08-09T05:16:48+00:00",
             "end_time": "2026-08-09T05:17:48+00:00",
             "attributes": {"b2e.harness": "claude_code",
                            "b2e.run.model_id": "claude-haiku-4-5-20251001",
                            "input.value": "сколько витрин?",
                            "output.value": "три",
                            "llm.token_count.prompt": 100}},
            {"context": {"span_id": "i1", "trace_id": trace_id}, "parent_id": "r",
             "name": "iteration.1", "span_kind": "CHAIN",
             "start_time": "2026-08-09T05:16:48+00:00",
             "end_time": "2026-08-09T05:17:00+00:00", "attributes": {}},
            {"context": {"span_id": "l1", "trace_id": trace_id}, "parent_id": "i1",
             "name": "llm.messages.create", "span_kind": "LLM",
             "start_time": "2026-08-09T05:16:49+00:00",
             "end_time": "2026-08-09T05:16:52+00:00",
             "attributes": {
                 "llm.token_count.prompt": 100,
                 "b2e.llm.ttft_ms": 1386,
                 "llm.output_messages.0.message.role": "assistant",
                 "llm.output_messages.0.message.contents.0.message_content.type":
                     "reasoning",
                 "llm.output_messages.0.message.contents.0.message_content.text":
                     "надо загрузить инструменты",
                 "llm.input_messages.0.message.role": "user",
                 "llm.input_messages.0.message.content": "сколько витрин?"}},
        ],
    }


def test_a_live_document_is_assembled_from_the_rest_shape():
    document = traceview.from_phoenix(_FakePhoenix([_rest_trace()]), limit=100)
    trace = document["traces"][0]
    assert trace["layer_counts"]["iteration_CHAIN"] == 1
    assert trace["reasoning"]["reasoning_texts"] == ["надо загрузить инструменты"]
    assert trace["user_prompt"] == "сколько витрин?"
    # The reconstruction is dropped from the live view and the caveat says so.
    assert any("input_messages is omitted" in c
               for c in document["_about"]["caveats"])


def test_the_live_document_keeps_prompts_when_asked():
    document = traceview.from_phoenix(_FakePhoenix([_rest_trace()]),
                                      include_input_messages=True)
    llm = document["traces"][0]["spans"][2]["attributes"]["llm"]
    assert llm["input_messages"][0]["message"]["content"] == "сколько витрин?"


def test_the_newest_turn_comes_first():
    older = _rest_trace("old")
    older["start_time"] = "2026-08-01T00:00:00+00:00"
    document = traceview.from_phoenix(_FakePhoenix([older, _rest_trace("new")]))
    assert [t["trace_id"] for t in document["traces"]] == ["new", "old"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_USER", AUTH[0])
    monkeypatch.setenv("ADMIN_PASSWORD", AUTH[1])
    monkeypatch.setenv("B2E_REGISTRY_DB", str(tmp_path / "registry.db"))
    state = AdminState()
    yield TestClient(create_app(state)), state
    state.registry.close()


def test_the_explorer_tab_is_in_the_navigation(client):
    c, _ = client
    assert 'href="explorer"' in c.get("/", auth=AUTH).text


def test_the_explorer_renders_the_viewer_with_the_data_embedded(client):
    c, state = client
    state._phoenix = _FakePhoenix([_rest_trace()])
    response = c.get("/explorer", auth=AUTH)
    assert response.status_code == 200
    assert "Agent trace explorer" in response.text
    assert "надо загрузить инструменты" in response.text
    assert traceview.PLACEHOLDER not in response.text
    assert "script-src 'unsafe-inline'" in response.headers["Content-Security-Policy"]


def test_the_explorer_requires_authentication(client):
    c, _ = client
    assert c.get("/explorer").status_code == 401
    assert c.get("/explorer.json").status_code == 401


def test_the_explorer_says_plainly_when_phoenix_is_down(client):
    """This page failing says something about Phoenix. A stack trace on an
    operator's screen says nothing they can act on."""
    c, state = client
    state._phoenix = _FakePhoenix(error=RuntimeError("connection refused"))
    response = c.get("/explorer", auth=AUTH)
    assert response.status_code == 200
    assert "Phoenix is not answering" in response.text
    assert "connection refused" in response.text


def test_an_empty_store_is_a_sentence_not_a_blank_viewer(client):
    c, state = client
    state._phoenix = _FakePhoenix([])
    assert "No traces recorded yet" in c.get("/explorer", auth=AUTH).text


def test_the_json_endpoint_returns_the_same_document(client):
    c, state = client
    state._phoenix = _FakePhoenix([_rest_trace()])
    body = c.get("/explorer.json", auth=AUTH).json()
    assert body["_analysis"]["corpus"]["traces"] == 1
    assert body["traces"][0]["reasoning"]["reasoning_blocks"] == 1
