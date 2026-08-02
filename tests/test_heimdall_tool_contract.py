"""The MCP tool descriptions are the agent's only manual. Keep them true.

This file exists because of a measured failure. Asked how many men and women
work at the company, the agent paged `employee_actual`, got 1000 rows, and
answered "618 women, 382 men, 1000 total" — for a corpus of 294 000. Pressed on
it, it said *«Ничего не выдумываю: API вернул ровно 1000 записей»*.

The API had been honest: the envelope carried `has_next_page: true`, and the
right answer was one aggregate call away (`metrics: ["fact_count"]` grouped by
`gender` → 88 093 / 143 788). But nothing the agent could read said so. The
`limit` and `offset` fields were bare `{"type": "integer"}` with no description,
no text anywhere mentioned `has_next_page` or the 1000 ceiling, and the skill
library shipped empty — so `get_docs`, `find_skills` and `get_overview` all
returned nothing. The agent was not ignoring the manual; there was no manual.

These tests pin the parts of the contract whose absence caused that.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

BRIDGE = pathlib.Path("heimdall/bridge.py")
SKILLS = pathlib.Path("heimdall-skills")


@pytest.fixture(scope="module")
def tools():
    """The TOOLS list, imported without starting the bridge's stdio loop."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bridge", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {t["name"]: t for t in module.TOOLS}


def _query_props(tools):
    return tools["mcp_query"]["inputSchema"]["properties"]


# ------------------------------------------------- the pagination contract


def test_mcp_query_warns_that_results_are_paginated(tools):
    text = tools["mcp_query"]["description"]
    assert "has_next_page" in text
    assert "1000" in text


def test_limit_documents_the_silent_clamp(tools):
    """The clamp is a deliberate trap (`silent_limit_clamp`). A trap should
    catch an agent that does not check its results — not one that was never told
    the ceiling exists."""
    text = _query_props(tools)["limit"]["description"]
    assert "1000" in text
    assert "молч" in text.lower(), "the silent part is the whole hazard"


def test_offset_explains_how_to_page(tools):
    text = _query_props(tools)["offset"]["description"]
    assert "has_next_page" in text
    assert "order_by" in text, "paging without a stable sort skips and repeats rows"


def test_every_query_field_is_described(tools):
    """A bare {"type": "integer"} tells the model the shape and nothing about
    the meaning, which is how `limit` came to be used as if it were a total."""
    undescribed = [name for name, spec in _query_props(tools).items()
                   if not spec.get("description")]
    assert not undescribed, undescribed


def test_counting_is_pointed_at_aggregation(tools):
    """The specific wrong turn: counting rows instead of asking for a metric."""
    text = tools["mcp_query"]["description"].lower()
    assert "metrics" in text and "агрегат" in text


def test_describe_model_separates_columns_from_metrics(tools):
    text = tools["describe_model"]["description"].lower()
    assert "metrics" in text and "columns" in text


def test_get_docs_names_the_topics_that_exist(tools):
    """A topic list in the description is what makes the library discoverable:
    `get_docs` with an unknown topic returns a stub, and an agent that gets a
    stub once does not ask twice."""
    text = tools["get_docs"]["description"]
    for topic in ("limits", "aggregate", "filters", "errors"):
        assert topic in text, topic


# --------------------------------------------------------- the library


def test_the_skill_library_is_not_empty():
    """It shipped with only a README, so every discovery tool returned nothing.
    An empty library is a legitimate *experimental condition*, but it is not a
    legitimate default."""
    docs = sorted(p.name for p in SKILLS.rglob("*.md") if p.name != "README.md")
    assert len(docs) >= 10, docs
    assert "limits.md" in docs and "aggregate.md" in docs


def test_the_library_covers_the_pagination_hazard():
    text = (SKILLS / "general" / "limits.md").read_text("utf-8")
    assert "has_next_page" in text
    assert "1000" in text


def test_every_reference_doc_has_loadable_frontmatter():
    """The registry keys skills off frontmatter; a doc without it is invisible
    to find_skills and get_docs however good its prose."""
    from heimdall.skills.registry import Registry

    registry = Registry.load(SKILLS)
    names = registry.all_names()
    assert len(names) >= 10
    assert not registry.duplicates()
    assert not registry.dangling_links(), registry.dangling_links()


def test_documented_examples_are_syntactically_valid():
    """Executing them needs a live emulator — `make check-docs` does that. This
    is the part that can run in CI: a malformed body teaches the agent a shape
    the server will reject."""
    fence = re.compile(r"```json[ \t]*\r?\n(.*?)```", re.S)
    for path in SKILLS.rglob("*.md"):
        for i, block in enumerate(fence.findall(path.read_text("utf-8"))):
            try:
                json.loads(block)
            except ValueError as exc:
                raise AssertionError(f"{path.name}#{i}: {exc}") from exc
