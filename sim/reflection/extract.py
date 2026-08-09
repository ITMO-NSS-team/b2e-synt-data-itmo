"""One turn's Phoenix spans, reduced to value-free call records.

``query_shape`` is the single most important function in Plan B, and it earns
that status by carrying two guarantees at once rather than one apiece:

* **Anti-memorisation.** Reflection reads traces and writes lessons that land in
  every future system prompt. If a lesson could carry a filter value, a person
  id or a paraphrase of a question, memory would stop teaching method and start
  transporting answers — the measured learning curve would then be
  memorisation, and the shared arm (three times the episodes) would look best
  for exactly the wrong reason.
* **Cross-employee privacy.** Three fleet instances act for three different
  employees with three different row-level permissions. Person ids, surnames
  and unit names live only in filter *values* and in ``output.value`` — never
  in a query's structure. If a value leaked through the shape, one employee's
  data would reach another employee's agent through the shared memory pack.

Neither guarantee can be dropped without breaking the other, which is why one
function carries both rather than two functions that could drift apart.

Where the real shapes came from
--------------------------------
The filter grammar below is read from ``heimdall/engine/filters.py:30-52``
(``_NODE_FIELDS``), not from the illustrative JSON in the plan — the plan's own
example uses a ``node``/``args`` shape the real grammar does not have (real
nodes use ``type``, and ``and``/``or`` carry ``conditions``, not ``args``).
Trusting the plan's prose over the code it describes would have produced a
walker that silently matches nothing.

The span shapes (which attribute holds a tool's raw parameters, which child
span is the Heimdall HTTP call, what a live ``condition_param`` filter actually
carries) were read off real spans pulled from the running Phoenix instance
(``b2e-sim-phoenix-1``, see ``tests/test_reflection_extract.py``'s property
test), not invented. That corpus is what caught the sharpest edge case here:
``condition_param`` nodes recorded from live agent turns carry a real
``person_id`` inside ``args`` (e.g. ``{"name": "oshs_owned", "args":
{"person_id": "<uuid>"}, ...}``) — a field the plan's node-field table lists
but its prose never calls out. Dropping ``args`` unconditionally, alongside
``value``/``pattern``/``expr``, is what closes that leak.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sim.research.evaluate import TraceFacts
from sim.traceview import attr, normalise_rest_span

# --------------------------------------------------------------- filter walk

#: Node types that terminate a branch with one comparison. Taken verbatim from
#: ``heimdall.engine.filters._NODE_FIELDS`` — the grammar's own registry —
#: rather than re-typed by hand, so a future node the filter engine grows is a
#: visible gap here (an unrecognised ``type`` yields no leaf, not a guess).
_LEAF_TYPES = frozenset({
    "condition", "condition_in", "condition_like", "condition_null",
    "condition_array", "condition_param",
})
#: ``and``/``or`` fan out over ``conditions``; ``not`` recurses into the single
#: ``condition`` it wraps. Three of the nine node types in the grammar; the
#: other six are the leaves above.
_LOGICAL_TYPES = frozenset({"and", "or"})


def _walk_filter(node: Any) -> list[tuple[str, str, str]]:
    """Every leaf of a filter AST as ``(node_type, column, operator)`` — no
    ``value``, no ``pattern``, no ``args``, no ``expr``.

    Recurses through the whole tree rather than a flat ``dict.items()`` scan:
    ``and``/``or``/``not`` nest arbitrarily, and a walker that only looked at
    the top level would silently lose every condition buried inside one of
    them — a lost node is a lesson keyed on a shape that never occurred, which
    is as bad as a leaked value for a different reason (it teaches the model a
    fiction about what queries look like).

    ``condition_param``'s ``column`` lives under the key ``name`` instead —
    the one field-name irregularity in the grammar (see
    ``heimdall/engine/filters.py:97``) — and its ``args`` dict is exactly the
    kind of place a value hides that a naive "keep column and operator, drop
    value" rule would miss; see the module docstring for the real example that
    proved it.

    Malformed or unrecognised input (an agent that guessed at the filter
    grammar and sent something structurally invalid) yields no leaves rather
    than raising: the point of this function is that it can never be the
    reason a value escapes, and a function that raises on bad input still has
    to be called from somewhere that decides what to do with the exception —
    "nothing leaks" should not depend on that call site getting it right.
    """
    if not isinstance(node, dict):
        return []
    ntype = node.get("type")
    if ntype in _LOGICAL_TYPES:
        children = node.get("conditions")
        if not isinstance(children, list):
            return []
        leaves: list[tuple[str, str, str]] = []
        for child in children:
            leaves.extend(_walk_filter(child))
        return leaves
    if ntype == "not":
        return _walk_filter(node.get("condition"))
    if ntype in _LEAF_TYPES:
        column = node.get("name") if ntype == "condition_param" else node.get("column")
        operator = node.get("operator")
        return [(ntype, str(column), str(operator))]
    return []


# ------------------------------------------------------------------ shape

def _limit_bucket(limit: Any) -> str:
    """Coarse bucket for a row cap. Buckets, not the number itself, because
    the number is exactly the kind of numeric literal the reflection guard
    (Task 4, rule G2) would otherwise have to police one more time — a bucket
    already carries the operationally relevant fact ("small page" vs. "the
    whole mart") without being a literal a lesson could quote back."""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return "none"
    if n <= 0:
        return "none"
    if n <= 10:
        return "1-10"
    if n <= 100:
        return "11-100"
    if n <= 1000:
        return "101-1000"
    return ">1000"


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted(str(v) for v in value))


def query_shape(params: dict[str, Any]) -> dict[str, Any]:
    """The value-free shape of one tool call's parameters.

    Only a fixed allow-list of structural keys is read — ``schema``,
    ``logic_model``, ``columns``, ``metrics``, ``filters``, ``order_by``,
    ``limit`` — and everything else in ``params`` (a Bash command, a
    ``find_skills`` free-text query, a ``get_docs`` topic) is silently
    ignored rather than merged in. That is deliberate defence in depth on top
    of the filter walker: a caller cannot leak a value through this function
    by adding a key the allow-list does not name, only by params legitimately
    using one of the seven structural keys above with a value inside it — and
    those seven are exactly the ones ``_walk_filter`` and the tuple helpers
    below strip.

    Column and metric *names* (``"person_id"``, ``"employee_full_name"``) are
    kept. They are catalog metadata — which field was asked for — not the
    values inside it; the same name already appears, harmlessly, as the
    ``column`` of a ``filter_nodes`` leaf. ``order_by`` entries are
    structural for the same reason: the grammar has no value-bearing field on
    an order clause.
    """
    params = params if isinstance(params, dict) else {}
    order_by_raw = params.get("order_by")
    order_by = tuple(
        (str(o.get("field")), str(o.get("kind")), str(o.get("direction")))
        for o in order_by_raw if isinstance(o, dict)
    ) if isinstance(order_by_raw, list) else ()

    return {
        "schema": params.get("schema"),
        "logic_model": params.get("logic_model"),
        "columns": _str_tuple(params.get("columns")),
        "metrics": _str_tuple(params.get("metrics")),
        "filter_nodes": _walk_filter(params.get("filters")),
        "order_by": order_by,
        "limit_bucket": _limit_bucket(params.get("limit")),
        "argument_keys": tuple(sorted(str(k) for k in params.keys())),
    }


# -------------------------------------------------------------- call records

@dataclass(frozen=True, slots=True)
class CallRecord:
    """One tool call, in the same value-free vocabulary ``query_shape``
    produces — this is the type callers actually build episode cards from."""

    seq: int
    tool: str
    schema: str | None
    logic_model: str | None
    columns: tuple[str, ...]
    filter_nodes: tuple[tuple[str, str, str], ...]
    order_by: tuple[tuple[str, str, str], ...]
    limit_bucket: str
    http_status: int | None
    error_code: str | None
    rows: int
    argument_keys: tuple[str, ...]


#: ``tool.mcp__heimdall__mcp_query`` (the TOOL span name Claude Code's harness
#: emits, see ``sim/telemetry.py:open_tool_span``) reduced to ``mcp_query``
#: (the short endpoint name ``sim/agent/tools.py`` and
#: ``heimdall/engine/errors.py`` both use). A non-Heimdall tool (``Bash``,
#: ``ToolSearch``) has no ``mcp__heimdall__`` prefix to strip and passes
#: through unchanged — it still becomes a ``CallRecord``, just one whose
#: Heimdall-specific fields are all empty, because a turn's shape includes
#: what else the agent tried, not only its Heimdall calls.
_TOOL_SPAN_PREFIX = "tool."
_MCP_PREFIX = "mcp__heimdall__"


def _short_tool(span_name: str) -> str:
    name = str(span_name or "").removeprefix(_TOOL_SPAN_PREFIX)
    return name.removeprefix(_MCP_PREFIX)


def _parsed_input(span: dict[str, Any]) -> dict[str, Any]:
    """A TOOL span's parameters, decoded from ``input.value``.

    That attribute is a JSON-encoded string (``sim.telemetry.set_attr``
    encodes every non-primitive value before it reaches the wire), not a
    nested object — ``normalise_rest_span``'s ``unflatten`` only reassembles
    dotted *keys*, it does not parse string *contents*. A tool call whose
    input cannot be recovered (missing, truncated, not an object) contributes
    no parameters rather than raising: a turn with one unreadable call should
    still yield a record for every other call in it.
    """
    raw = attr(span, "input", "value")
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def call_records(spans: list[dict[str, Any]]) -> list[CallRecord]:
    """Every TOOL call in one turn, paired with its Heimdall CHAIN child.

    Spans arrive in Phoenix's flattened REST shape and are normalised here
    (not by the caller) so this function is the one place that shape decision
    is made for the whole reflection pipeline — the same reason
    ``sim.traceview`` gives for owning ``normalise_rest_span`` centrally.

    Pairing is structural: a Heimdall CHAIN span whose ``parent_id`` is a
    TOOL span's id belongs to it (``sim/telemetry.py``'s ``start_heimdall_call``
    opens that span as the current span inside the tool's own call, which is
    exactly a parent/child relationship). At most one such child is expected
    per tool call under this harness's architecture (one MCP call issues at
    most one HTTP request); if more than one were ever found, the earliest by
    start time is used and the rest ignored, because pairing to the wrong one
    is worse than pairing to none. A TOOL span with no matching child (a Bash
    call, a describe_model whose Heimdall span fell outside the fetched
    window) still produces a record — with ``http_status=None`` rather than a
    guessed value — because its *shape* (what was asked for) is still evidence
    even when its *outcome* is not observed.
    """
    normed = [normalise_rest_span(s) for s in spans]
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for span in normed:
        parent = span.get("parent_id")
        if parent and span.get("span_kind") == "CHAIN" and \
                str(span.get("name") or "").startswith("heimdall."):
            by_parent.setdefault(parent, []).append(span)
    for children in by_parent.values():
        children.sort(key=lambda s: s.get("_start_epoch") or 0)

    tool_spans = [s for s in normed if s.get("span_kind") == "TOOL"]
    tool_spans.sort(key=lambda s: s.get("_start_epoch") or 0)

    records: list[CallRecord] = []
    for seq, span in enumerate(tool_spans):
        params = _parsed_input(span)
        shape = query_shape(params)
        heimdall_children = by_parent.get(span.get("span_id"), [])
        heimdall = heimdall_children[0] if heimdall_children else None

        http_status = attr(heimdall, "b2e", "http", "status") if heimdall else None
        error_code = attr(heimdall, "b2e", "heimdall", "error_code") if heimdall else None
        rows = attr(heimdall, "b2e", "heimdall", "rows") if heimdall else None

        records.append(CallRecord(
            seq=seq,
            tool=_short_tool(span.get("name")),
            schema=shape["schema"],
            logic_model=shape["logic_model"],
            columns=shape["columns"],
            filter_nodes=tuple(shape["filter_nodes"]),
            order_by=shape["order_by"],
            limit_bucket=shape["limit_bucket"],
            http_status=int(http_status) if http_status is not None else None,
            error_code=str(error_code) if error_code else None,
            rows=int(rows) if rows is not None else 0,
            argument_keys=shape["argument_keys"],
        ))
    return records


# ---------------------------------------------------------------- turn facts

def turn_facts(spans: list[dict[str, Any]]) -> TraceFacts:
    """Reduce one turn's spans to the fields Plan A's scorer needs.

    ``spans`` is one turn's worth (one trace — the root ``b2e.turn`` AGENT
    span and everything under it), matching what ``call_records`` expects. A
    caller holding a whole session's spans across several turns groups them by
    trace id before calling either function; grouping is not this function's
    job because it has no opinion about what a "turn" boundary is beyond "one
    root span" — that definition belongs to whoever owns epoch/turn semantics
    (Plan C's driver).
    """
    normed = [normalise_rest_span(s) for s in spans]
    root = next((s for s in normed if s.get("span_kind") == "AGENT"
                and not s.get("parent_id")), None)
    records = call_records(spans)

    http_statuses = tuple(c.http_status for c in records if c.http_status is not None)
    heimdall_calls = sum(
        1 for s in normed
        if s.get("span_kind") == "CHAIN" and str(s.get("name") or "").startswith("heimdall.")
    )
    error_codes = tuple(c.error_code for c in records if c.error_code)
    # Restricted to mcp_query: describe_model/list_models always report
    # rows=0 and have no "columns of a mart projection" to speak of, so mixing
    # them in would let a turn that only ever called describe_model read as
    # "queried the mart and got zero rows back" for the no_data category, and
    # would let their spurious 0-column entries set the over-fetch baseline's
    # floor artificially low for api_validity.
    query_records = [c for c in records if c.tool == "mcp_query"]
    rows_returned = tuple(c.rows for c in query_records if c.http_status is not None)
    successful_columns = tuple(
        len(c.columns) for c in query_records if c.http_status == 200 and c.rows > 0
    )
    columns_requested = tuple(len(c.columns) for c in query_records)

    repeated_calls, pagination_walks = _waste_counts(spans)

    token_count = attr(root, "llm", "token_count") if root else None
    tokens = int((token_count or {}).get("total") or 0) if isinstance(token_count, dict) else 0
    seconds = round((root.get("duration_ms") or 0) / 1000, 3) if root else 0.0
    memory_tokens = attr(root, "b2e", "memory", "tokens") if root else None

    return TraceFacts(
        http_statuses=http_statuses,
        heimdall_calls=heimdall_calls,
        rows_returned=rows_returned,
        error_codes=error_codes,
        repeated_calls=repeated_calls,
        pagination_walks=pagination_walks,
        columns_requested=columns_requested,
        tokens=tokens,
        seconds=seconds,
        memory_tokens=int(memory_tokens) if memory_tokens is not None else 0,
        successful_columns=successful_columns,
    )


def _waste_counts(spans: list[dict[str, Any]]) -> tuple[int, int]:
    """Byte-identical repeats, and offset-incrementing pagination walks.

    Unlike ``query_shape``, this reads the *raw* parameters — including their
    values — because it never leaves this process: the result is two integers
    for ``TraceFacts``, not text that could reach a prompt. Detecting "the
    same call twice" or "a paging walk" genuinely needs the values (two
    ``employee_id IN [...]`` calls with different ids are not a repeat), so
    stripping them first would make both counts meaningless.
    """
    normed = [normalise_rest_span(s) for s in spans]
    tool_spans = [s for s in normed if s.get("span_kind") == "TOOL"]
    tool_spans.sort(key=lambda s: s.get("_start_epoch") or 0)

    seen: set[str] = set()
    repeated = 0
    pagination = 0
    previous: dict[str, Any] | None = None
    for span in tool_spans:
        raw = attr(span, "input", "value")
        text = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True, default=str)
        if text in seen:
            repeated += 1
        seen.add(text)

        params = _parsed_input(span)
        if previous is not None and _is_pagination_step(previous, params):
            pagination += 1
        previous = params
    return repeated, pagination


def _is_pagination_step(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Same call in every respect except an ``offset`` that moved forward."""
    before_offset, after_offset = before.get("offset"), after.get("offset")
    if before_offset is None and after_offset is None:
        return False
    try:
        if not (int(after_offset or 0) > int(before_offset or 0)):
            return False
    except (TypeError, ValueError):
        return False
    rest_before = {k: v for k, v in before.items() if k != "offset"}
    rest_after = {k: v for k, v in after.items() if k != "offset"}
    return rest_before == rest_after
