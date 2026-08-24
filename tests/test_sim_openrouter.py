"""OpenRouter adapter: Anthropic loop shape on an OpenAI-compatible wire.

No network. A fake httpx response is enough to prove the translation the live
client will send and the loop will consume.
"""
from __future__ import annotations

import json

from sim.agent.config import DEFAULT_OPENROUTER_MODEL
from sim.agent.llm import LLMResponse, build_client
from sim.agent.openrouter import (
    OpenRouterClient,
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    openai_finish_to_stop_reason,
    openai_message_to_anthropic_content,
)
from sim.agent.shipped import OPENROUTER_CONFIG_REF, shipped_configs
from sim.agent.tools import tool_schemas


def test_anthropic_tools_become_openai_functions():
    schemas = tool_schemas(("mcp_query",))
    converted = anthropic_tools_to_openai(schemas)
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "mcp_query"
    assert converted[0]["function"]["parameters"]["type"] == "object"
    assert "schema" in converted[0]["function"]["parameters"]["properties"]


def test_ox_alpha_requests_low_reasoning_effort(monkeypatch):
    """Ox Alpha defaults to effort=max and cannot send none; low leaves budget
    for tool_calls instead of burning max_tokens on hidden thinking."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    client = OpenRouterClient()
    fake = _FakeHttp(_OK_BODY)
    client._http = fake
    _complete(client)
    assert fake.calls[0]["json"]["reasoning"] == {"effort": "low"}


def test_reasoning_effort_env_overrides_ox_alpha_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "high")
    client = OpenRouterClient()
    fake = _FakeHttp(_OK_BODY)
    client._http = fake
    _complete(client)
    assert fake.calls[0]["json"]["reasoning"] == {"effort": "high"}


def test_assistant_round_trip_keeps_reasoning_for_the_next_turn():
    messages = [{
        "role": "assistant",
        "content": [
            {"type": "reasoning", "text": "need list_models",
             "details": [{"type": "reasoning.summary", "text": "need list_models"}]},
            {"type": "tool_use", "id": "c1", "name": "list_models", "input": {}},
        ],
    }]
    openai_messages = anthropic_messages_to_openai("", messages)
    assistant = openai_messages[0]
    assert assistant["reasoning"] == "need list_models"
    assert assistant["reasoning_details"] == [
        {"type": "reasoning.summary", "text": "need list_models"}]
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["content"] is None


def test_tool_use_round_trip_through_openai_shape():
    system = "you are the agent"
    messages = [
        {"role": "user", "content": "Сколько сотрудников в моём подразделении?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_1", "name": "mcp_query",
             "input": {"schema": "dm_core", "logic_model": "employee_actual"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1",
             "content": '{"n": 21}'},
        ]},
    ]
    openai_messages = anthropic_messages_to_openai(system, messages)
    assert openai_messages[0] == {"role": "system", "content": system}
    assert openai_messages[1]["role"] == "user"
    assistant = openai_messages[2]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]
                      )["schema"] == "dm_core"
    assert openai_messages[3] == {
        "role": "tool", "tool_call_id": "call_1", "content": '{"n": 21}',
    }


def test_openai_tool_call_becomes_llmresponse_tool_uses():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_9",
            "type": "function",
            "function": {
                "name": "find_skills",
                "arguments": '{"query": "численность"}',
            },
        }],
    }
    content = openai_message_to_anthropic_content(message)
    response = LLMResponse(
        content=content, stop_reason=openai_finish_to_stop_reason("tool_calls"),
        prompt_tokens=10, completion_tokens=4)
    uses = response.tool_uses()
    assert response.stop_reason == "tool_use"
    assert uses == [{
        "type": "tool_use", "id": "call_9", "name": "find_skills",
        "input": {"query": "численность"},
    }]


def test_empty_content_promotes_reasoning_to_visible_text():
    """Ox Alpha spent the completion on reasoning and left content empty.
    Without this, the agent returns answer:\"\" after a successful turn."""
    content = openai_message_to_anthropic_content({
        "role": "assistant",
        "content": None,
        "reasoning": "В подразделении 12 сотрудников.",
    })
    types = [b["type"] for b in content]
    assert "reasoning" in types
    assert "text" in types
    response = LLMResponse(
        content=content, stop_reason="end_turn",
        prompt_tokens=10, completion_tokens=50)
    assert response.text() == "В подразделении 12 сотрудников."
    assert response.tool_uses() == []


def test_reasoning_is_not_promoted_when_the_model_called_a_tool():
    content = openai_message_to_anthropic_content({
        "role": "assistant",
        "content": "",
        "reasoning": "надо вызвать list_models",
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "list_models", "arguments": "{}"},
        }],
    })
    assert [b["type"] for b in content] == ["reasoning", "tool_use"]
    assert LLMResponse(content=content, stop_reason="tool_use",
                       prompt_tokens=1, completion_tokens=1).text() == ""


def test_free_model_projects_zero_cost():
    response = LLMResponse(
        content=[{"type": "text", "text": "21"}], stop_reason="end_turn",
        prompt_tokens=1000, completion_tokens=1000)
    assert response.cost_usd(DEFAULT_OPENROUTER_MODEL) == 0.0
    assert response.cost_usd("qwen/qwen3-8b:free") == 0.0
    assert response.cost_usd("stealth/ox-alpha") == 0.0
    assert response.cost_usd("claude-haiku-4-5-20251001") > 0


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, json):
        self.calls.append({"url": url, "json": json})
        return _FakeResponse(self.payload)


class _SequenceHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json):
        self.calls.append({"url": url, "json": json})
        payload, status = self.responses.pop(0)
        return _FakeResponse(payload, status_code=status)


_RATE_LIMIT_BODY = {
    "error": {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {
            "raw": "openai/gpt-oss-20b:free is temporarily rate-limited upstream.",
            "provider_name": "Darkbloom",
            "retry_after_seconds": 30,
            "headers": {"Retry-After": "30"},
        },
    }
}

_OK_BODY = {
    "choices": [{
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "21"},
    }],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


def _complete(client):
    return client.complete(
        model=DEFAULT_OPENROUTER_MODEL, system="sys",
        messages=[{"role": "user", "content": "сколько?"}],
        tools=[], temperature=0.0, max_tokens=64)


def test_openrouter_client_complete_uses_translated_payload(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = OpenRouterClient()
    fake = _FakeHttp({
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "В подразделении 21 человек."},
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    })
    client._http = fake
    response = client.complete(
        model=DEFAULT_OPENROUTER_MODEL, system="sys",
        messages=[{"role": "user", "content": "сколько?"}],
        tools=tool_schemas(("mcp_query",)), temperature=0.0, max_tokens=256)
    sent = fake.calls[0]["json"]
    assert sent["model"] == DEFAULT_OPENROUTER_MODEL
    assert sent["messages"][0]["role"] == "system"
    assert sent["tools"][0]["function"]["name"] == "mcp_query"
    assert response.text() == "В подразделении 21 человек."
    assert response.stop_reason == "end_turn"
    assert response.tool_uses() == []
    assert response.prompt_tokens == 12


def test_build_client_live_prefers_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = build_client("live", "cassettes")
    assert isinstance(client, OpenRouterClient)


def test_build_client_live_anthropic_when_provider_forced(monkeypatch):
    from sim.agent import llm as llm_mod

    sentinel = object()
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(llm_mod, "AnthropicClient", lambda **kwargs: sentinel)
    assert llm_mod.build_client("live", "cassettes") is sentinel


def test_shipped_openrouter_config_is_messages_api():
    _kind, body, _note = shipped_configs()[OPENROUTER_CONFIG_REF]
    from sim.agent.config import AgentConfig

    config = AgentConfig.from_dict(body)
    assert config.harness == "messages_api"
    assert config.model_id == DEFAULT_OPENROUTER_MODEL
    assert config.max_output_tokens == 16384
    assert config.conversation_mode == "stateless"


def test_openrouter_429_waits_and_retries_the_same_provider(monkeypatch):
    """Free models often have one upstream. Ignoring it 404s; wait instead."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_IGNORE_PROVIDERS", raising=False)
    client = OpenRouterClient()
    fake = _SequenceHttp([(_RATE_LIMIT_BODY, 429), (_OK_BODY, 200)])
    sleeps: list[float] = []
    client._http = fake
    client._sleep = sleeps.append
    assert _complete(client).text() == "21"
    assert sleeps == [30.0]
    assert "provider" not in fake.calls[1]["json"]


def test_openrouter_404_ignored_providers_retries_without_ignore(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_IGNORE_PROVIDERS", "Darkbloom")
    client = OpenRouterClient()
    ignored_body = {
        "error": {
            "message": "All providers have been ignored. To change your "
                       "default ignored providers, visit: "
                       "https://openrouter.ai/settings/privacy",
            "code": 404,
        }
    }
    fake = _SequenceHttp([(ignored_body, 404), (_OK_BODY, 200)])
    sleeps: list[float] = []
    client._http = fake
    client._sleep = sleeps.append
    assert _complete(client).text() == "21"
    assert "ignore" in fake.calls[0]["json"]["provider"]
    assert "provider" not in fake.calls[1]["json"]


def test_openrouter_429_gives_up_after_attempts(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MAX_ATTEMPTS", "2")
    monkeypatch.delenv("OPENROUTER_IGNORE_PROVIDERS", raising=False)
    client = OpenRouterClient()
    fake = _SequenceHttp([(_RATE_LIMIT_BODY, 429), (_RATE_LIMIT_BODY, 429)])
    client._http = fake
    client._sleep = lambda _s: None
    try:
        _complete(client)
    except RuntimeError as exc:
        assert "429" in str(exc)
        assert "Darkbloom" in str(exc) or "rate-limited" in str(exc)
    else:
        raise AssertionError("expected RuntimeError after exhausted 429s")
