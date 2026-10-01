"""Both CLI harnesses receive the same bounded, lossless catalog pages."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from heimdall import bridge
from heimdall.app import DESCRIPTION_MAX_BYTES, _describe, _install_error_handler, _mcp_v1_router
from heimdall.engine.errors import fail
from sim.agent.claude_code import ClaudeCodeHarness
from sim.agent.config import AgentConfig
from sim.agent.opencode import OpenCodeHarness


@pytest.fixture
def model():
    def member(name, parameters=None):
        return SimpleNamespace(name=name, type="String", description="Описание " * 80,
                               parameters=parameters)
    return SimpleNamespace(
        schema="dm_core", logic_model="large", summary="Large model", is_history=False,
        columns={f"col_{i}": member(f"col_{i}") for i in range(642)},
        metrics={"fact_count": member("fact_count"),
                 "at_date": member("at_date", [{"name": "date", "type": "Date32"}])},
        time_dimensions={"date": SimpleNamespace(name="date", granularities=["day", "month"])},
    )


@pytest.fixture
def client(model):
    def lookup(schema, name, channel):
        if schema == "forbidden":
            raise fail("forbidden", "No access")
        return model
    app = FastAPI()
    _install_error_handler(app)
    app.include_router(_mcp_v1_router(SimpleNamespace(model=lookup)))
    return TestClient(app)


def test_large_description_pages_recover_all_members_without_truncation(model):
    offset = 0
    recovered = {kind: [] for kind in ("columns", "metrics", "time_dimensions")}
    while True:
        page = _describe(model, offset=offset, limit=100)
        assert len(json.dumps(page, ensure_ascii=False).encode("utf-8")) <= DESCRIPTION_MAX_BYTES
        for kind in recovered:
            recovered[kind].extend(page[kind])
        paging = page["pagination"]
        assert paging["total_members"] == 645
        assert paging["returned_members"] > 0
        if not paging["has_next_page"]:
            assert paging["next_offset"] is None
            break
        assert paging["next_offset"] > offset
        offset = paging["next_offset"]
    assert [m["name"] for m in recovered["columns"]] == list(model.columns)
    assert [m["name"] for m in recovered["metrics"]] == list(model.metrics)
    assert recovered["metrics"][1]["parameters"] == model.metrics["at_date"].parameters
    assert recovered["time_dimensions"][0]["granularities"] == ["day", "month"]
    assert all(m["description"] == model.columns[m["name"]].description
               for m in recovered["columns"])


def test_search_section_and_end_of_pages(model):
    page = _describe(model, section="metrics", search="FACT_COUNT")
    assert [m["name"] for m in page["metrics"]] == ["fact_count"]
    assert not page["columns"] and not page["time_dimensions"]
    assert page["pagination"]["total_members"] == 1
    assert not page["pagination"]["has_next_page"]
    assert not _describe(model, offset=999)["pagination"]["has_next_page"]
    assert _describe(model, search="not-a-member")["pagination"]["total_members"] == 0


@pytest.mark.parametrize("params", [{"offset": -1}, {"limit": 0}, {"limit": 101},
                                   {"section": "secrets"}])
def test_invalid_page_requests_are_rejected(client, params):
    response = client.get("/api/v1/mcp/models/dm_core/large/", params=params,
                          headers={"Authorization": "Bearer test"})
    assert response.status_code == 422


def test_both_harnesses_use_identical_paged_bridge_without_file_access(client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    description = next(tool for tool in bridge.TOOLS if tool["name"] == "describe_model")
    assert {"offset", "limit", "section", "search"} <= description["inputSchema"]["properties"].keys()
    assert "pagination.next_offset" in description["description"]
    requests = []

    def http(method, path, *, params=None, body=None):
        requests.append(params)
        response = client.request(method, path, params=params,
                                  headers={"Authorization": "Bearer test"})
        return response.status_code, response.json()

    monkeypatch.setattr(bridge, "_http", http)
    kwargs = dict(heimdall_url="http://unused", heimdall_token="test",
                  bridge_path="/app/heimdall/bridge.py")
    config = AgentConfig(tool_subset=("describe_model",))
    claude = ClaudeCodeHarness(**kwargs)
    opencode = OpenCodeHarness(**kwargs)
    cc_mcp = claude.mcp_config("42", config, "/tmp/log")["mcpServers"]["heimdall"]
    oc_mcp = opencode.build_config(config=config, system_prompt="system", employee_id="42",
                                    trace_log="/tmp/log")["mcp"]["heimdall"]
    assert cc_mcp["args"] == oc_mcp["command"][1:]
    assert cc_mcp["env"]["HEIMDALL_TOOL_SUBSET"] == oc_mcp["environment"]["HEIMDALL_TOOL_SUBSET"]
    assert "Read" in claude.denied_tools(config)
    assert "read" not in opencode._permissions(config)
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "describe_model", "arguments": {"schema": "dm_core", "logic_model": "large",
        "offset": 0, "limit": 5, "section": "metrics", "search": "fact_count"}}}
    first = bridge.handle(request)
    second = bridge.handle(request)
    assert first == second
    payload = json.loads(first["result"]["content"][0]["text"])
    assert [m["name"] for m in payload["metrics"]] == ["fact_count"]
    assert requests == [dict(offset=0, limit=5, section="metrics", search="fact_count")] * 2
    assert len(first["result"]["content"][0]["text"].encode("utf-8")) <= DESCRIPTION_MAX_BYTES


def test_pagination_preserves_access_errors(client, monkeypatch):
    response = client.get("/api/v1/mcp/models/forbidden/large/",
                          params={"offset": 2, "search": "fact_count"},
                          headers={"Authorization": "Bearer test"})
    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"
    assert "pagination" not in response.json()


def test_single_oversized_member_fails_explicitly_instead_of_losing_content(model, client):
    model.columns["col_0"].description = "x" * 20_000
    response = client.get("/api/v1/mcp/models/dm_core/large/",
                          headers={"Authorization": "Bearer test"})
    assert response.status_code == 400
    assert response.json()["code"] == "request-validation-error"
    assert "превышает бюджет" in response.json()["detail"]
