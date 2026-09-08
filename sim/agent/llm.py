"""Model access, with a replay mode so CI and integration tests cost nothing.

Implementations behind one interface:

``AnthropicClient``   real calls against Anthropic, OAuth token or API key
``RecordingClient``   a real client that also writes a cassette
``ReplayClient``      serves cassettes; makes no network call and spends nothing

The cassette key is a SHA-256 over the *semantic* request — model, system prompt,
messages, tools, temperature, max tokens — so an unrelated change (a new session
id, a timestamp) does not invalidate every recording, while a change that would
genuinely alter the model's answer does.

``make test`` runs in replay. A replay miss is an error, never a silent fallback
to the network: a test suite that quietly starts spending money is worse than one
that fails.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

#: Claude Code subscription tokens are OAuth bearer tokens, not API keys. The
#: SDK sends them as `Authorization: Bearer …` via `auth_token=`, and the beta
#: header below is what the subscription endpoint expects. Both are config, not
#: constants, because this is an integration whose contract we do not own.
OAUTH_BETA_HEADER = "oauth-2025-04-20"

#: Price table for cost projection, USD per million tokens. Anthropic does not
#: return a price with the response, so every cost figure in this system is a
#: projection from these rates — stated plainly rather than presented as measured.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
DEFAULT_PRICE = (1.00, 5.00)


class ReplayMiss(RuntimeError):
    """No cassette for this request. Never falls back to a live call."""


@dataclass(slots=True)
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    remaining_token_budget: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content
                       if b.get("type") == "text")

    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    def cost_usd(self, model: str) -> float:
        if model.startswith("glm-"):
            prompt_rate, completion_rate = (0.0, 0.0)
        else:
            prompt_rate, completion_rate = PRICES_USD_PER_MTOK.get(model, DEFAULT_PRICE)
        return (self.prompt_tokens * prompt_rate
                + self.completion_tokens * completion_rate) / 1_000_000


class LLMClient(Protocol):
    mode: str

    def complete(self, *, model: str, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]], temperature: float,
                 max_tokens: int) -> LLMResponse: ...


# ------------------------------------------------------------------ cassettes


def cassette_key(*, model: str, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]], temperature: float,
                 max_tokens: int) -> str:
    payload = json.dumps({
        "model": model, "system": system, "messages": messages,
        "tools": [t.get("name") for t in tools],
        "temperature": temperature, "max_tokens": max_tokens,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _response_to_dict(r: LLMResponse) -> dict[str, Any]:
    return {
        "content": r.content, "stop_reason": r.stop_reason,
        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "cache_read_tokens": r.cache_read_tokens,
        "cache_write_tokens": r.cache_write_tokens,
        "remaining_token_budget": r.remaining_token_budget,
    }


def _response_from_dict(d: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=d["content"], stop_reason=d.get("stop_reason"),
        prompt_tokens=int(d.get("prompt_tokens", 0)),
        completion_tokens=int(d.get("completion_tokens", 0)),
        cache_read_tokens=int(d.get("cache_read_tokens", 0)),
        cache_write_tokens=int(d.get("cache_write_tokens", 0)),
        remaining_token_budget=d.get("remaining_token_budget"),
    )


# ------------------------------------------------------------------- clients


class ReplayClient:
    """Serves recorded responses. Makes no network call, spends nothing."""

    mode = "replay"

    def __init__(self, cassette_dir: str | Path) -> None:
        self.dir = Path(cassette_dir)

    def complete(self, **kwargs: Any) -> LLMResponse:
        key = cassette_key(**kwargs)
        path = self.dir / f"{key}.json"
        if not path.exists():
            raise ReplayMiss(
                f"no cassette {key[:16]}… in {self.dir}. Record it with "
                f"B2E_LLM_MODE=record, or fix the request so it matches an "
                f"existing recording. Replay never falls back to a live call."
            )
        return _response_from_dict(json.loads(path.read_text("utf-8"))["response"])


class AnthropicClient:
    """Real calls against the Anthropic API.

    Authorises with the subscription OAuth token when one is present, falling
    back to an API key. The token is read from the environment and is never
    written to a file, a log, or a span.
    """

    mode = "live"

    def __init__(self, *, oauth_token: str | None = None,
                 api_key: str | None = None, timeout: float = 120.0) -> None:
        import anthropic

        oauth_token = oauth_token or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or None
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or None
        if not oauth_token and not api_key:
            raise RuntimeError(
                "no credentials: set CLAUDE_CODE_OAUTH_TOKEN (subscription) or "
                "ANTHROPIC_API_KEY. Both are read from the environment only."
            )

        if oauth_token:
            self._client = anthropic.Anthropic(
                auth_token=oauth_token, timeout=timeout,
                default_headers={"anthropic-beta": OAUTH_BETA_HEADER},
            )
        else:
            self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def complete(self, *, model: str, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]], temperature: float,
                 max_tokens: int) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model, "system": system, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        message = self._client.messages.create(**kwargs)
        usage = message.usage
        content = [
            b.model_dump() if hasattr(b, "model_dump") else dict(b)
            for b in message.content
        ]
        return LLMResponse(
            content=content,
            stop_reason=message.stop_reason,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )


class RecordingClient:
    """A live client that writes a cassette for every call."""

    mode = "record"

    def __init__(self, inner: LLMClient, cassette_dir: str | Path) -> None:
        self.inner = inner
        self.dir = Path(cassette_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def complete(self, **kwargs: Any) -> LLMResponse:
        response = self.inner.complete(**kwargs)
        key = cassette_key(**kwargs)
        (self.dir / f"{key}.json").write_text(json.dumps({
            "recorded_at": time.time(),
            "request": {k: v for k, v in kwargs.items() if k != "messages"},
            "response": _response_to_dict(response),
        }, ensure_ascii=False, indent=1), "utf-8")
        return response


class ScriptedClient:
    """Returns a fixed sequence of responses. For unit tests only.

    Distinct from ReplayClient on purpose: replay is keyed by request content and
    is used for integration tests that must exercise the real request-shaping
    code. Scripted ignores the request and is used where the point of the test is
    the loop's control flow.
    """

    mode = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise ReplayMiss("ScriptedClient ran out of responses")
        return self._responses.pop(0)


def _live_client() -> LLMClient:
    return AnthropicClient()


def build_client(mode: str, cassette_dir: str | Path) -> LLMClient:
    """Factory driven by ``B2E_LLM_MODE``."""
    if mode == "replay":
        return ReplayClient(cassette_dir)
    if mode == "record":
        return RecordingClient(_live_client(), cassette_dir)
    if mode == "live":
        return _live_client()
    raise ValueError(f"unknown LLM mode {mode!r}; expected replay | record | live")
