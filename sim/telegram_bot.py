"""Telegram bot — a pass-through surface for researchers.

Deliberately thin. The spec is explicit: "no additional logics in it – just
by-passing to B2E". So this holds exactly one piece of state, the mapping from a
Telegram chat to a B2E session, and otherwise forwards text in and text out.

That thinness is a measurement property, not laziness. Any retry, any rewording,
any "helpful" summarisation here would appear in the traces as agent behaviour
and quietly contaminate every comparison a researcher makes between manual and
batch runs.

Long polling rather than webhooks: a webhook needs a public HTTPS endpoint for
Telegram to call, which would mean another hole in the reverse proxy. Polling
keeps the bot strictly outbound, so it adds no exposed surface at all.

No external dependency — the Bot API is plain HTTPS/JSON, and adding a library
for three endpoints would be a poor trade on this box.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger("b2e.telegram")

API = "https://api.telegram.org/bot{token}/{method}"

HELP = (
    "B2E — исследовательский стенд.\n\n"
    "Просто напишите вопрос: он уйдёт в агента как есть.\n\n"
    "/new — начать новую сессию\n"
    "/whoami — какая личность сотрудника сейчас используется\n"
    "/employee <id> — переключить личность\n"
)


class TelegramBridge:
    """Chat id -> B2E session. Nothing else is remembered here."""

    def __init__(self, *, token: str, agent_url: str, default_employee: str) -> None:
        if not token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is unset. It is read from the environment "
                "only and is never written to a file or a commit."
            )
        self.token = token
        self.agent_url = agent_url.rstrip("/")
        self.default_employee = default_employee
        self.sessions: dict[int, str] = {}
        self.employees: dict[int, str] = {}
        # trust_env is left ON here, unlike internal clients: reaching
        # api.telegram.org may legitimately require the host's proxy.
        self._tg = httpx.Client(timeout=70.0)
        self._agent = httpx.Client(timeout=180.0, trust_env=False)

    # ------------------------------------------------------------ telegram

    def _call(self, method: str, **payload: Any) -> dict[str, Any]:
        response = self._tg.post(API.format(token=self.token, method=method),
                                 json=payload)
        return response.json()

    def send(self, chat_id: int, text: str) -> None:
        # Telegram rejects messages over 4096 characters. Chunking is transport,
        # not logic: the text is not altered, only split.
        for i in range(0, len(text) or 1, 4000):
            self._call("sendMessage", chat_id=chat_id,
                       text=text[i:i + 4000] or "(пустой ответ)")

    # --------------------------------------------------------------- agent

    def ensure_session(self, chat_id: int) -> str:
        if chat_id in self.sessions:
            return self.sessions[chat_id]
        employee = self.employees.get(chat_id, self.default_employee)
        response = self._agent.post(f"{self.agent_url}/sessions",
                                    json={"employee_id": employee})
        response.raise_for_status()
        session_id = response.json()["session_id"]
        self.sessions[chat_id] = session_id
        return session_id

    def ask(self, chat_id: int, text: str) -> str:
        session_id = self.ensure_session(chat_id)
        response = self._agent.post(
            f"{self.agent_url}/sessions/{session_id}/messages",
            json={"content": text})
        if response.status_code >= 400:
            return f"[b2e {response.status_code}] {response.text[:600]}"
        body = response.json()
        stats = body.get("stats", {})
        # The footer is diagnostics for a researcher, appended after the answer.
        # It never changes the question or the answer.
        footer = (f"\n\n— {stats.get('heimdall_calls', 0)} вызовов API, "
                  f"{stats.get('total_tokens', 0)} токенов, "
                  f"trace {str(body.get('trace_id'))[:16]}")
        return (body.get("answer") or "(пустой ответ)") + footer

    # ---------------------------------------------------------------- loop

    def handle(self, message: dict[str, Any]) -> None:
        chat_id = message["chat"]["id"]
        text = (message.get("text") or "").strip()
        if not text:
            return

        if text.startswith("/start") or text.startswith("/help"):
            self.send(chat_id, HELP)
            return
        if text.startswith("/new"):
            self.sessions.pop(chat_id, None)
            self.send(chat_id, "Новая сессия открыта.")
            return
        if text.startswith("/whoami"):
            employee = self.employees.get(chat_id, self.default_employee)
            self.send(chat_id, f"employee_id = {employee}\n"
                               f"session = {self.sessions.get(chat_id, '—')}")
            return
        if text.startswith("/employee"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                self.send(chat_id, "Использование: /employee <employee_id>")
                return
            self.employees[chat_id] = parts[1].strip()
            self.sessions.pop(chat_id, None)
            self.send(chat_id, f"Личность переключена на {parts[1].strip()}.")
            return

        try:
            self.send(chat_id, self.ask(chat_id, text))
        except httpx.HTTPError as exc:
            self.send(chat_id, f"[transport] {exc}")

    def run(self) -> None:
        log.info("telegram bridge polling, agent=%s", self.agent_url)
        offset = 0
        while True:
            try:
                payload = self._call("getUpdates", offset=offset, timeout=60)
            except httpx.HTTPError as exc:
                log.warning("getUpdates failed: %s", exc)
                time.sleep(5)
                continue
            if not payload.get("ok"):
                log.warning("telegram error: %s", payload)
                time.sleep(5)
                continue
            for update in payload.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message") or update.get("edited_message")
                if message:
                    try:
                        self.handle(message)
                    except Exception:                        # pragma: no cover
                        log.exception("handler failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    TelegramBridge(
        token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        agent_url=os.environ.get("B2E_AGENT_URL", "http://b2e-agent:8082"),
        default_employee=os.environ.get("TELEGRAM_DEFAULT_EMPLOYEE", ""),
    ).run()


if __name__ == "__main__":
    main()
