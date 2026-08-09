"""Span extraction and the privacy boundary.

``query_shape`` carries two guarantees that cannot be tested separately
without missing the point: it is both the anti-memorisation boundary (no
value can become a lesson) and the cross-employee privacy boundary (no value
can leak from one employee's trace into another's shared memory). The first
two tests below are the load-bearing ones — they use the *real* filter
grammar (``heimdall/engine/filters.py``'s ``type``/``conditions``/``condition``
node shape, not the plan's illustrative ``node``/``args`` prose) and would
fail against an implementation that iterates ``dict.items()`` instead of
walking the tree, or that forgets ``condition_param``'s ``args``.

The property test at the bottom runs the same function over real spans
recorded by the live agent fleet today, pulled from the running Phoenix
instance and cached in ``.pytest-reflection-corpus/`` (gitignored, per
``.gitignore``'s ``.pytest-*/`` pattern — never committed). It is skipped, not
failed, when that cache or the full ``data/truth/people.json`` snapshot is
absent, because both are local-environment artefacts no CI checkout can be
expected to have.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sim.reflection import extract

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ fixtures


def _span(kind: str, name: str, span_id: str, parent: str | None,
         attributes: dict | None = None, *,
         start: str = "2026-08-09T05:16:48.000000+00:00",
         end: str = "2026-08-09T05:16:49.000000+00:00") -> dict:
    """One raw REST span, in Phoenix's own flattened-attribute shape — the
    same shape ``PhoenixClient.spans_for_session`` returns, so these fixtures
    exercise ``normalise_rest_span`` for real rather than assuming it away."""
    return {
        "context": {"span_id": span_id, "trace_id": "t1"},
        "parent_id": parent, "name": name, "span_kind": kind,
        "start_time": start, "end_time": end,
        "attributes": attributes or {},
    }


def _tool_span(span_id: str, parent: str, params: dict, *,
              tool: str = "mcp__heimdall__mcp_query",
              start: str = "2026-08-09T05:16:48.000000+00:00",
              end: str = "2026-08-09T05:16:49.000000+00:00") -> dict:
    return _span("TOOL", f"tool.{tool}", span_id, parent, {
        "input.value": json.dumps(params, ensure_ascii=False),
        "input.mime_type": "application/json",
        "tool.name": tool,
    }, start=start, end=end)


def _heimdall_span(span_id: str, parent: str, *, endpoint: str = "mcp_query",
                   status: int = 200, rows: int = 3,
                   error_code: str | None = None,
                   start: str = "2026-08-09T05:16:48.100000+00:00",
                   end: str = "2026-08-09T05:16:48.900000+00:00") -> dict:
    attrs = {
        "b2e.http.method": "POST", "b2e.http.path": "/api/v1/mcp/query/",
        "b2e.heimdall.endpoint": endpoint,
        "b2e.http.status": status, "b2e.heimdall.rows": rows,
    }
    if error_code:
        attrs["b2e.heimdall.error_code"] = error_code
    return _span("CHAIN", f"heimdall.{endpoint}", span_id, parent, attrs,
                start=start, end=end)


# ---------------------------------------------------------------- query_shape


def test_query_shape_discards_every_filter_value():
    """Person ids, surnames and unit names live only in filter values and in
    output.value. If they survive this function, a shared memory becomes a
    channel for one employee's data to reach another's agent.

    Uses the real grammar (``type``/``conditions``), not the plan's
    illustrative ``node``/``args`` shape — the real one is what
    ``heimdall/engine/filters.py:30-52`` actually defines and what spans on
    the wire actually carry.
    """
    params = {
        "schema": "dm_core", "logic_model": "employee_actual",
        "columns": ["grade_level"],
        "filters": {"type": "and", "conditions": [
            {"type": "condition", "column": "person_id", "operator": "=",
             "value": "8f14e45f-ceea-467a-9d0f-2b4c3a1e5b77"},
            {"type": "condition_like", "column": "employee_full_name",
             "operator": "LIKE", "pattern": "Иванов%"},
        ]},
    }
    shape = extract.query_shape(params)
    blob = json.dumps(shape, ensure_ascii=False)
    assert "8f14e45f" not in blob
    assert "Иванов" not in blob
    assert ("condition", "person_id", "=") in [tuple(n) for n in shape["filter_nodes"]]
    assert ("condition_like", "employee_full_name", "LIKE") in \
        [tuple(n) for n in shape["filter_nodes"]]


def test_query_shape_walks_the_whole_filter_ast_not_a_flat_dict():
    """The filter body is a nine-node tree. An implementation iterating
    dict.items() at the top level silently loses everything nested inside
    and/or/not, and a lost node is a lesson keyed on a shape that never
    occurred."""
    params = {"schema": "dm_core", "logic_model": "employee_actual",
             "filters": {"type": "and", "conditions": [
                 {"type": "or", "conditions": [
                     {"type": "condition", "column": "grade_level",
                      "operator": ">=", "value": 8},
                     {"type": "not", "condition": {
                         "type": "condition_null", "column": "unit_id",
                         "operator": "IS NULL"}},
                 ]},
                 {"type": "condition_in", "column": "employee_id",
                  "operator": "IN", "value": ["1", "2"]},
             ]}}
    shape = extract.query_shape(params)
    leaves = {tuple(n) for n in shape["filter_nodes"]}
    assert ("condition", "grade_level", ">=") in leaves
    assert ("condition_null", "unit_id", "IS NULL") in leaves
    assert ("condition_in", "employee_id", "IN") in leaves
    assert len(leaves) == 3


def test_query_shape_drops_condition_param_args_even_when_they_carry_a_value():
    """condition_param addresses a parametric column via ``name`` (not
    ``column``) and its ``args`` dict is exactly where a real person_id was
    found hiding on a live recorded trace (see the module docstring) — a rule
    that only strips ``value``/``pattern`` and forgets ``args`` would pass this
    one specific shape straight through."""
    params = {"schema": "dm_core", "logic_model": "employee_actual",
             "filters": {"type": "condition_param", "name": "oshs_owned",
                         "args": {"person_id": "288402be-e848-44cb-a296-2a9e2100506b"},
                         "operator": "=", "value": 1}}
    shape = extract.query_shape(params)
    blob = json.dumps(shape, ensure_ascii=False)
    assert "288402be" not in blob
    assert ("condition_param", "oshs_owned", "=") in [tuple(n) for n in shape["filter_nodes"]]


def test_query_shape_ignores_a_malformed_filter_body_rather_than_leaking_it():
    """An agent that guesses at the filter grammar sends structurally invalid
    bodies (observed on the live corpus: ``{"position_name": "руководитель"}``
    with no ``type`` at all). The real service rejects this outright; this
    function must not accidentally launder it into a "clean" shape by falling
    back to copying keys."""
    params = {"schema": "dm_core", "logic_model": "employee_actual",
             "filters": {"position_name": "руководитель"}}
    shape = extract.query_shape(params)
    blob = json.dumps(shape, ensure_ascii=False)
    assert "руководитель" not in blob
    assert shape["filter_nodes"] == []


def test_query_shape_ignores_unknown_top_level_keys():
    """A find_skills free-text query or a get_docs topic could carry
    anything, including a name. Neither key is in the structural allow-list,
    so neither should reach the shape at all."""
    params = {"query": "уволить Иванова", "topic": "премия Смирновой", "limit": 10}
    shape = extract.query_shape(params)
    blob = json.dumps(shape, ensure_ascii=False)
    assert "Иванов" not in blob and "Смирнов" not in blob


@pytest.mark.parametrize("limit,expected", [
    (None, "none"), (0, "none"), (5, "1-10"), (10, "1-10"),
    (11, "11-100"), (100, "11-100"), (101, "101-1000"), (1000, "101-1000"),
    (1001, ">1000"), (1_000_000, ">1000"),
])
def test_limit_bucket_boundaries(limit, expected):
    params = {"schema": "s", "logic_model": "m"}
    if limit is not None:
        params["limit"] = limit
    assert extract.query_shape(params)["limit_bucket"] == expected


# --------------------------------------------------------------- call_records


def test_call_records_pairs_a_tool_span_with_its_heimdall_child():
    tool = _tool_span("tool1", "iter1", {"schema": "dm_core",
                                         "logic_model": "employee_actual",
                                         "columns": ["grade_level"], "limit": 5})
    heimdall = _heimdall_span("h1", "tool1", status=200, rows=3)
    records = extract.call_records([tool, heimdall])
    assert len(records) == 1
    record = records[0]
    assert record.tool == "mcp_query"
    assert record.schema == "dm_core" and record.logic_model == "employee_actual"
    assert record.http_status == 200
    assert record.rows == 3
    assert record.error_code is None
    assert record.limit_bucket == "1-10"
    assert record.seq == 0


def test_call_records_reports_none_status_for_an_unmatched_tool_span():
    """A TOOL span whose child is unmatched keeps its params but reports
    http_status=None — an absent observation must not be reported as a
    guessed status."""
    bash = _tool_span("tool1", "iter1", {"command": "echo hi"}, tool="Bash")
    records = extract.call_records([bash])
    assert len(records) == 1
    assert records[0].tool == "Bash"
    assert records[0].http_status is None
    assert records[0].rows == 0


def test_call_records_carries_the_error_code_from_a_failed_call():
    tool = _tool_span("tool1", "iter1", {"schema": "anagent",
                                         "logic_model": "employee_actual_orion"})
    heimdall = _heimdall_span("h1", "tool1", status=403, rows=0, error_code="forbidden")
    records = extract.call_records([tool, heimdall])
    assert records[0].http_status == 403
    assert records[0].error_code == "forbidden"


def test_call_records_orders_calls_by_start_time_not_input_order():
    early = _tool_span("tool_early", "iter1", {"schema": "s", "logic_model": "m1"},
                       start="2026-08-09T05:16:48.000000+00:00",
                       end="2026-08-09T05:16:48.500000+00:00")
    late = _tool_span("tool_late", "iter1", {"schema": "s", "logic_model": "m2"},
                      start="2026-08-09T05:16:49.000000+00:00",
                      end="2026-08-09T05:16:49.500000+00:00")
    records = extract.call_records([late, early])
    assert [r.logic_model for r in records] == ["m1", "m2"]
    assert [r.seq for r in records] == [0, 1]


# ----------------------------------------------------------------- turn_facts


def _turn_spans():
    root = _span("AGENT", "b2e.turn", "root", None, {
        "llm.token_count.total": 1200, "llm.token_count.prompt": 1000,
        "llm.token_count.completion": 200, "b2e.memory.tokens": 150,
    }, start="2026-08-09T05:16:47.000000+00:00",
       end="2026-08-09T05:16:50.000000+00:00")
    tool1 = _tool_span("tool1", "root", {"schema": "dm_core",
                                         "logic_model": "employee_actual",
                                         "columns": ["grade_level"]})
    heimdall1 = _heimdall_span("h1", "tool1", status=200, rows=5)
    tool2 = _tool_span("tool2", "root", {"schema": "dm_core",
                                         "logic_model": "employee_actual",
                                         "metrics": ["fact_count"]},
                       start="2026-08-09T05:16:49.000000+00:00",
                       end="2026-08-09T05:16:49.500000+00:00")
    heimdall2 = _heimdall_span("h2", "tool2", status=403, rows=0,
                               error_code="forbidden",
                               start="2026-08-09T05:16:49.100000+00:00",
                               end="2026-08-09T05:16:49.400000+00:00")
    return [root, tool1, heimdall1, tool2, heimdall2]


def test_turn_facts_reads_tokens_and_seconds_off_the_root_span():
    facts = extract.turn_facts(_turn_spans())
    assert facts.tokens == 1200
    assert facts.memory_tokens == 150
    assert facts.seconds == pytest.approx(3.0, abs=0.01)


def test_turn_facts_counts_heimdall_calls_and_error_codes():
    facts = extract.turn_facts(_turn_spans())
    assert facts.heimdall_calls == 2
    assert facts.http_statuses == (200, 403)
    assert facts.error_codes == ("forbidden",)
    assert facts.rows_returned == (5, 0)
    assert facts.columns_requested == (1, 0)


def test_turn_facts_memory_tokens_defaults_to_zero_without_the_attribute():
    """Correct rather than merely tolerated: an arm with no memory block
    subtracts nothing (see TraceFacts's own docstring)."""
    spans = [s for s in _turn_spans() if s.get("name") != "b2e.turn"]
    root = _span("AGENT", "b2e.turn", "root", None, {},
                start="2026-08-09T05:16:47.000000+00:00",
                end="2026-08-09T05:16:50.000000+00:00")
    facts = extract.turn_facts([root] + spans)
    assert facts.memory_tokens == 0


def test_turn_facts_detects_a_byte_identical_repeated_call():
    tool1 = _tool_span("t1", "root", {"schema": "s", "logic_model": "m",
                                      "columns": ["a"]})
    tool2 = _tool_span("t2", "root", {"schema": "s", "logic_model": "m",
                                      "columns": ["a"]},
                       start="2026-08-09T05:16:49.000000+00:00",
                       end="2026-08-09T05:16:49.500000+00:00")
    root = _span("AGENT", "b2e.turn", "root", None, {},
                start="2026-08-09T05:16:47.000000+00:00",
                end="2026-08-09T05:16:50.000000+00:00")
    facts = extract.turn_facts([root, tool1, tool2])
    assert facts.repeated_calls == 1


def test_turn_facts_detects_an_offset_incrementing_pagination_walk():
    tool1 = _tool_span("t1", "root", {"schema": "s", "logic_model": "m",
                                      "columns": ["a"], "limit": 100, "offset": 0})
    tool2 = _tool_span("t2", "root", {"schema": "s", "logic_model": "m",
                                      "columns": ["a"], "limit": 100, "offset": 100},
                       start="2026-08-09T05:16:49.000000+00:00",
                       end="2026-08-09T05:16:49.500000+00:00")
    root = _span("AGENT", "b2e.turn", "root", None, {},
                start="2026-08-09T05:16:47.000000+00:00",
                end="2026-08-09T05:16:50.000000+00:00")
    facts = extract.turn_facts([root, tool1, tool2])
    assert facts.pagination_walks == 1
    assert facts.repeated_calls == 0


# -------------------------------------------------------- real-corpus property


CORPUS_FILE = ROOT / ".pytest-reflection-corpus" / "raw_spans.json"
PEOPLE_FILE = ROOT / "data" / "truth" / "people.json"

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_CYRILLIC_WORD_RE = re.compile(r"[А-ЯЁ][а-яё]+")


def test_query_shape_leaks_no_real_person_id_or_surname_over_recorded_calls():
    """Property test over real spans (plan Task 3, Step 4): no person_id from
    truth/people.json and no surname from the snapshot's dictionary survives
    query_shape for any recorded call.

    The corpus is real TOOL-span parameters pulled from the live b2e-sim
    Phoenix instance (see the module docstring for how) — genuine values an
    agent actually sent, not values invented for this test. Skipped, not
    failed, when either local artefact this needs (the cached corpus, the
    full population snapshot) is absent — both are gitignored and specific to
    this environment, and a property test that can prove nothing without them
    must say so rather than report a false pass.
    """
    if not CORPUS_FILE.exists():
        pytest.skip(f"no recorded-span corpus at {CORPUS_FILE}; this is a "
                    f"local, gitignored artefact — see the module docstring "
                    f"for how it was built")
    if not PEOPLE_FILE.exists():
        pytest.skip(f"no population snapshot at {PEOPLE_FILE} to check "
                    f"person_ids against (data/ is gitignored)")

    from b2e.gen.names import build_surnames

    people = json.loads(PEOPLE_FILE.read_text("utf-8"))
    person_ids = set(people["person_id"])
    surnames: set[str] = set()
    for masculine, feminine in build_surnames():
        surnames.add(masculine)
        surnames.add(feminine)

    raw_spans = json.loads(CORPUS_FILE.read_text("utf-8"))
    checked_calls = 0
    calls_with_real_pii = 0
    for span in raw_spans:
        if span.get("span_kind") != "TOOL":
            continue
        raw_input = (span.get("attributes") or {}).get("input.value")
        if not isinstance(raw_input, str):
            continue
        try:
            params = json.loads(raw_input)
        except ValueError:
            continue
        if not isinstance(params, dict):
            continue
        checked_calls += 1

        real_ids = set(_UUID_RE.findall(raw_input)) & person_ids
        real_names = set(_CYRILLIC_WORD_RE.findall(raw_input)) & surnames
        if real_ids or real_names:
            calls_with_real_pii += 1

        shape = extract.query_shape(params)
        blob = json.dumps(shape, ensure_ascii=False)
        for person_id in real_ids:
            assert person_id not in blob, (
                f"real person_id {person_id} survived query_shape for a "
                f"recorded call: {raw_input[:200]}")
        for name in real_names:
            assert name not in blob, (
                f"real surname {name} survived query_shape for a recorded "
                f"call: {raw_input[:200]}")

    # Corpus sanity: a property test that never actually exercised a call
    # carrying real PII would pass vacuously and prove nothing about the
    # boundary it claims to check.
    assert checked_calls >= 50, "corpus too small to be a meaningful property check"
    assert calls_with_real_pii >= 1, (
        "no recorded call in the corpus carried a real person_id or surname; "
        "this test cannot demonstrate the boundary holds")
