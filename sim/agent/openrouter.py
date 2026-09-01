"""OpenRouter client: OpenAI-compatible wire, Anthropic-shaped loop.

The agent loop (``sim.agent.loop``) speaks Anthropic tool-use blocks. OpenRouter
speaks OpenAI chat-completions. This module is the translation, and only the
translation — behaviour of the loop, fingerprint, and tool surface does not
change when the provider does.

Free models on OpenRouter rotate. ``openrouter/free`` is the documented router
that filters for the features a request actually uses (tool calling, in our
case). Pin a specific ``:free`` slug in admin UI if a run must name one model.

A 429 from a ``:free`` shared pool is retried by sleeping ``Retry-After``
(capped) against the same model. The named upstream is not ignored: free
slugs often have one provider, and skipping it 404s. The model id on the
request does not change.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from sim.agent.llm import LLMResponse

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Free-model shared pools 429 often. Sleep at most this long per retry so a
#: turn stays inside the HTTP client timeout rather than hanging the worker.
_MAX_RETRY_WAIT_S = 35.0
_DEFAULT_MAX_ATTEMPTS = 6
_SHARED_POOL_RETRY_FLOOR_S = 15.0


def _openrouter_proxy() -> str | None:
    """Proxy only this client's OpenRouter calls, never the whole agent.

    ``OPENROUTER_HTTP_PROXY`` is the campus tunnel (``openrouter-proxy.md``).
    Falling back to ``HTTPS_PROXY`` keeps the old VPS AGENT_HTTP_PROXY path.
    Empty means direct egress. The ITMO proxy must not see Heimdall/Phoenix.
    """
    return (os.environ.get("OPENROUTER_HTTP_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY")
            or None)


def anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic ``{name, description, input_schema}`` → OpenAI ``tools``."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {
                    "type": "object", "properties": {},
                },
            },
        })
    return converted


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def anthropic_messages_to_openai(
    system: str, messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten Anthropic content-block history into OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        role = message.get("role") or "user"
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue
        if role == "assistant":
            out.append(_assistant_blocks_to_openai(content))
            continue
        # Tool results travel as a user message of tool_result blocks in the
        # Anthropic loop; OpenAI wants one ``role=tool`` message per result.
        tool_results = [b for b in content if b.get("type") == "tool_result"]
        other = [b for b in content if b.get("type") != "tool_result"]
        for block in tool_results:
            out.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id") or "",
                "content": block.get("content") or "",
            })
        if other:
            text = "".join(
                b.get("text", "") for b in other if b.get("type") == "text")
            if text:
                out.append({"role": "user", "content": text})
    return out


def _assistant_blocks_to_openai(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    reasoning = "\n".join(
        b.get("text", "") for b in blocks
        if b.get("type") == "reasoning" and b.get("text"))
    details = next((b.get("details") for b in blocks
                    if b.get("type") == "reasoning" and b.get("details")), None)
    tool_calls = []
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        tool_calls.append({
            "id": block.get("id") or "",
            "type": "function",
            "function": {
                "name": block.get("name") or "",
                "arguments": json.dumps(block.get("input") or {},
                                        ensure_ascii=False),
            },
        })
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning"] = reasoning
    if details:
        message["reasoning_details"] = details
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not text:
            message["content"] = None
    return message


def openai_message_to_anthropic_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI assistant message → Anthropic content blocks the loop already reads.

    Reasoning models (Ox Alpha) often spend the completion budget on
    ``reasoning`` / ``reasoning_content`` and leave ``content`` empty. The
    loop's visible answer is ``LLMResponse.text()``, which only reads ``text``
    blocks, so an empty ``content`` becomes ``answer: ""`` after a successful
    turn. Promote leftover reasoning to a text block when there is nothing
    else to show and no tool call.
    """
    blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text") or ""})
            elif isinstance(part, str) and part:
                blocks.append({"type": "text", "text": part})
    reasoning = _reasoning_text(message)
    if reasoning:
        block: dict[str, Any] = {"type": "reasoning", "text": reasoning}
        if message.get("reasoning_details"):
            block["details"] = message["reasoning_details"]
        blocks.append(block)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or "",
            "name": function.get("name") or "",
            "input": _parse_arguments(function.get("arguments")),
        })
    has_text = any(b.get("type") == "text" and b.get("text") for b in blocks)
    has_tools = any(b.get("type") == "tool_use" for b in blocks)
    if reasoning and not has_text and not has_tools:
        blocks.append({"type": "text", "text": reasoning})
    return blocks


def _reasoning_text(message: dict[str, Any]) -> str:
    """Pull a visible string out of OpenRouter/OpenAI reasoning fields."""
    parts: list[str] = []
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, dict):
            for inner in ("text", "content", "summary"):
                piece = value.get(inner)
                if isinstance(piece, str) and piece.strip():
                    parts.append(piece.strip())
                    break
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            for inner in ("text", "summary", "content"):
                piece = item.get(inner)
                if isinstance(piece, str) and piece.strip():
                    parts.append(piece.strip())
                    break
    return "\n".join(parts)


def openai_finish_to_stop_reason(finish: str | None) -> str:
    if finish == "tool_calls":
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    if finish in ("error", "network_error"):
        return "error"
    return "end_turn"


def _choice_finish(choice: dict[str, Any]) -> str | None:
    raw = choice.get("finish_reason") or choice.get("native_finish_reason")
    return str(raw) if raw else None


def _is_empty_upstream_stop(
    choice: dict[str, Any], message: dict[str, Any],
    usage: dict[str, Any], content: list[dict[str, Any]],
) -> bool:
    """Stealth reports some upstream failures as HTTP 200 with no output.

    Observed on ox-alpha after a tool round: ``finish_reason=stop``, empty
    ``message``, ``prompt_tokens=0``, ``completion_tokens=0``, ~1s latency.
    Treating that as ``end_turn`` made the demo return ``answer: ""`` after
    Heimdall had already been called. Retry instead.
    """
    if content:
        return False
    if message.get("tool_calls") or _reasoning_text(message) or message.get(
            "reasoning_details"):
        return False
    finish = _choice_finish(choice)
    native = str(choice.get("native_finish_reason") or "")
    if finish in ("error", "network_error") or native in ("error", "network_error"):
        return True
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return prompt == 0 and completion == 0


def _empty_retry_wait(attempt: int) -> float:
    return min(5.0 * (2 ** attempt), _MAX_RETRY_WAIT_S)


def _error_object(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    return error if isinstance(error, dict) else body


def _all_providers_ignored(status: int, body: Any) -> bool:
    if status != 404:
        return False
    message = str(_error_object(body).get("message") or "")
    return "All providers have been ignored" in message


def _retry_after_seconds(response: Any, body: Any) -> float:
    headers = getattr(response, "headers", None) or {}
    raw = None
    if hasattr(headers, "get"):
        raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        metadata = _error_object(body).get("metadata") or {}
        if isinstance(metadata, dict):
            raw = metadata.get("retry_after_seconds")
            if raw is None:
                nested = metadata.get("headers") or {}
                if isinstance(nested, dict):
                    raw = nested.get("Retry-After") or nested.get("retry-after")
    try:
        wait = float(raw)
    except (TypeError, ValueError):
        wait = 5.0
    wait = min(max(wait, 1.0), _MAX_RETRY_WAIT_S)
    metadata = _error_object(body).get("metadata") or {}
    if isinstance(metadata, dict) and (
            metadata.get("limit_source") == "upstream_provider_shared_pool"):
        wait = max(wait, _SHARED_POOL_RETRY_FLOOR_S)
    return wait


def _env_ignore_providers() -> list[str]:
    raw = os.environ.get("OPENROUTER_IGNORE_PROVIDERS") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _max_attempts() -> int:
    raw = os.environ.get("OPENROUTER_MAX_ATTEMPTS") or str(_DEFAULT_MAX_ATTEMPTS)
    try:
        return max(1, min(int(raw), 8))
    except ValueError:
        return _DEFAULT_MAX_ATTEMPTS


def _reasoning_request(model: str) -> dict[str, Any] | None:
    """Ox Alpha cannot disable reasoning and defaults to ``effort=max``.

    That spends the whole ``max_tokens`` budget on hidden thinking, so the
    response arrives with empty ``content``, no ``tool_calls``, and
    ``answer: ""``. ``low`` is the minimum the endpoint accepts. Override with
    ``OPENROUTER_REASONING_EFFORT`` (``low`` / ``high`` / ``max`` / ``omit``).
    """
    raw = (os.environ.get("OPENROUTER_REASONING_EFFORT") or "").strip().lower()
    if raw in ("omit", "off", "none"):
        return None
    if raw in ("low", "high", "max"):
        return {"effort": raw}
    if model.startswith("stealth/") or "ox-alpha" in model:
        return {"effort": "low"}
    return None


class OpenRouterClient:
    """Live calls against OpenRouter's OpenAI-compatible Chat Completions API."""

    mode = "live"

    def __init__(self, *, api_key: str | None = None, timeout: float = 120.0,
                 base_url: str = OPENROUTER_URL) -> None:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or None
        if not api_key:
            raise RuntimeError(
                "no credentials: set OPENROUTER_API_KEY. "
                "The key is read from the environment only.")
        proxy = _openrouter_proxy()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("PUBLIC_URL", "https://localhost:8443"),
            "X-Title": os.environ.get("PHOENIX_PROJECT", "b2e-itmo"),
        }
        self._http = httpx.Client(
            timeout=timeout, trust_env=False, proxy=proxy, headers=headers)
        self._url = base_url
        self._sleep = time.sleep

    def complete(self, *, model: str, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]], temperature: float,
                 max_tokens: int) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages_to_openai(system, messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = anthropic_tools_to_openai(tools)
            payload["tool_choice"] = "auto"
        reasoning = _reasoning_request(model)
        if reasoning:
            payload["reasoning"] = reasoning

        ignored = list(_env_ignore_providers())
        last_error: Any = None
        last_status = 0
        for attempt in range(_max_attempts()):
            sent = dict(payload)
            # Do not auto-ignore a 429 provider. Free models often have one
            # upstream; ignoring it yields 404 "All providers have been ignored".
            if ignored:
                sent["provider"] = {
                    "allow_fallbacks": True,
                    "ignore": list(ignored),
                }

            response = self._http.post(self._url, json=sent)
            try:
                body = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"OpenRouter returned non-JSON ({response.status_code}): "
                    f"{response.text[:500]}") from exc

            error = body.get("error") if isinstance(body, dict) else body
            if response.status_code == 429:
                last_status, last_error = response.status_code, error
                if attempt + 1 >= _max_attempts():
                    break
                self._sleep(_retry_after_seconds(response, body))
                continue

            if _all_providers_ignored(response.status_code, body):
                last_status, last_error = response.status_code, error
                ignored = []
                if attempt + 1 >= _max_attempts():
                    break
                self._sleep(_retry_after_seconds(response, body))
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenRouter {response.status_code}: {error}")

            # Stealth sometimes wraps an upstream failure in HTTP 200 + error.
            if isinstance(error, dict) and (error.get("message") or error.get("code")):
                last_status, last_error = response.status_code, error
                if attempt + 1 >= _max_attempts():
                    break
                self._sleep(_retry_after_seconds(response, body))
                continue

            choices = body.get("choices") or []
            if not choices:
                last_status, last_error = response.status_code, (
                    "no choices in 200 response")
                if attempt + 1 >= _max_attempts():
                    break
                self._sleep(_empty_retry_wait(attempt))
                continue
            choice = choices[0]
            message = dict(choice.get("message") or {})
            if not message.get("reasoning") and choice.get("reasoning"):
                message["reasoning"] = choice["reasoning"]
            content = openai_message_to_anthropic_content(message)
            usage = body.get("usage") or {}
            if _is_empty_upstream_stop(choice, message, usage, content):
                reason = (
                    str(choice.get("native_finish_reason") or "")
                    or _choice_finish(choice)
                    or "upstream network_error")
                last_status, last_error = 200, f"empty completion ({reason})"
                if attempt + 1 >= _max_attempts():
                    break
                self._sleep(_empty_retry_wait(attempt))
                continue
            return LLMResponse(
                content=content,
                stop_reason=openai_finish_to_stop_reason(_choice_finish(choice)),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                raw=body,
            )

        raise RuntimeError(f"OpenRouter {last_status}: {last_error}")
