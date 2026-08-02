"""Telegram bridge — command dispatch and session continuity.

The bot had no tests, which was defensible while it was three `startswith`
branches over a stateless pass-through. It stopped being defensible once a chat
became a conversation: the difference between "this message continues the
session" and "this message resets it" is now the difference between the agent
having context and not, and that is exactly the class of bug nobody notices
until a transcript reads strangely a week later.

Nothing here talks to Telegram or to the agent. `TelegramBridge` is driven
directly and its two HTTP clients are replaced with recorders, so the assertions
are about dispatch and state rather than about httpx.
"""
from __future__ import annotations

from typing import Any

import pytest

from sim.telegram_bot import NEW_SESSION_COMMANDS, TelegramBridge


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected {self.status_code}")


class FakeAgent:
    """Stands in for the b2e-agent service, and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.session_counter = 0

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.calls.append((url, json))
        if url.endswith("/sessions"):
            self.session_counter += 1
            return FakeResponse({"session_id": f"ses_{self.session_counter}"})
        return FakeResponse({"answer": "ответ", "trace_id": "t" * 32,
                             "stats": {"heimdall_calls": 2, "total_tokens": 10}})


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.sent.append(json.get("text", ""))
        return FakeResponse({"ok": True})


@pytest.fixture(autouse=True)
def _no_host_proxy(monkeypatch):
    """The Telegram client keeps `trust_env=True` on purpose — reaching
    api.telegram.org from this host genuinely needs the proxy. That makes
    construction depend on the developer's environment: a SOCKS `ALL_PROXY`
    makes httpx raise for a missing `socksio` before a single assertion runs.
    Cleared here rather than worked around in the bot, because the production
    behaviour is the one we want to keep.
    """
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                 "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def bridge():
    b = TelegramBridge(token="tok", agent_url="http://b2e-agent:8082",
                       default_employee="2457060")
    b._tg = FakeTelegram()
    b._agent = FakeAgent()
    return b


def _say(bridge, text: str, chat_id: int = 7) -> None:
    bridge.handle({"chat": {"id": chat_id}, "text": text})


def _sessions_opened(bridge) -> list[dict[str, Any]]:
    return [body for url, body in bridge._agent.calls if url.endswith("/sessions")]


# --------------------------------------------------------- continuity


def test_consecutive_messages_share_one_session(bridge):
    """The whole point of the change: a follow-up must reach the same session,
    or the agent meets every question cold and "как я говорил выше" is a lie."""
    _say(bridge, "сколько человек в отделе?")
    _say(bridge, "а из них на удалёнке?")
    assert len(_sessions_opened(bridge)) == 1

    asked = [url for url, _ in bridge._agent.calls if url.endswith("/messages")]
    assert len(asked) == 2
    assert asked[0] == asked[1]


def test_separate_chats_never_share_a_session(bridge):
    """Two researchers in two chats are two conversations."""
    _say(bridge, "вопрос", chat_id=1)
    _say(bridge, "вопрос", chat_id=2)
    assert len(_sessions_opened(bridge)) == 2


def test_sessions_open_against_the_conversational_config(bridge):
    """`agent_config` is the stateless batch condition; a chat that opened a
    session against it would reset context on every message while still looking
    like a dialogue from the outside."""
    _say(bridge, "вопрос")
    assert _sessions_opened(bridge)[0]["config_ref"] == "agent_config_interactive"


def test_the_config_ref_is_overridable():
    b = TelegramBridge(token="tok", agent_url="http://x", default_employee="1",
                       config_ref="agent_config")
    b._tg, b._agent = FakeTelegram(), FakeAgent()
    _say(b, "вопрос")
    assert _sessions_opened(b)[0]["config_ref"] == "agent_config"


# ----------------------------------------------------------- resetting


@pytest.mark.parametrize("command", NEW_SESSION_COMMANDS)
def test_start_new_session_drops_the_context(bridge, command):
    _say(bridge, "первый вопрос")
    _say(bridge, command)
    _say(bridge, "второй вопрос")

    opened = _sessions_opened(bridge)
    assert len(opened) == 2, "the reset must force a genuinely new session"

    asked = [url for url, _ in bridge._agent.calls if url.endswith("/messages")]
    assert asked[0] != asked[1]


def test_start_new_session_is_not_forwarded_to_the_agent(bridge):
    _say(bridge, "/start_new_session")
    assert not [url for url, _ in bridge._agent.calls if url.endswith("/messages")]


def test_resetting_before_anything_started_is_not_an_error(bridge):
    _say(bridge, "/start_new_session")
    assert bridge._tg.sent, "the researcher must be told something happened"


def test_start_still_prints_help_and_does_not_reset(bridge):
    """`/start` is what Telegram sends when a chat is first opened."""
    _say(bridge, "/start")
    assert "B2E" in bridge._tg.sent[0]
    assert not _sessions_opened(bridge)


def test_help_advertises_the_reset_command(bridge):
    _say(bridge, "/help")
    assert "/start_new_session" in bridge._tg.sent[0]


# ------------------------------------------------- command boundaries


def test_a_question_beginning_like_a_command_is_still_a_question(bridge):
    """`startswith("/new")` matched `/newcomers`, so a question about new hires
    silently reset the session and was never asked."""
    _say(bridge, "/newcomers за квартал?")
    assert [url for url, _ in bridge._agent.calls if url.endswith("/messages")]


def test_commands_survive_the_at_botname_suffix(bridge):
    """In a group chat Telegram appends @botname to every command."""
    _say(bridge, "первый вопрос")
    _say(bridge, "/start_new_session@b2e_bot")
    _say(bridge, "второй вопрос")
    assert len(_sessions_opened(bridge)) == 2


def test_switching_employee_resets_the_session(bridge):
    """A session carries a Heimdall permission scope. Keeping it across an
    identity switch would answer as the previous employee."""
    _say(bridge, "вопрос")
    _say(bridge, "/employee 999")
    _say(bridge, "вопрос")

    opened = _sessions_opened(bridge)
    assert len(opened) == 2
    assert opened[1]["employee_id"] == "999"


def test_whoami_reports_the_config_ref(bridge):
    """Which config a chat is on decides whether it has memory at all, so it
    belongs in the one command that answers "what am I talking to"."""
    _say(bridge, "/whoami")
    assert "agent_config_interactive" in bridge._tg.sent[0]


def test_empty_messages_are_ignored(bridge):
    bridge.handle({"chat": {"id": 7}, "text": "   "})
    bridge.handle({"chat": {"id": 7}})
    assert not bridge._agent.calls
