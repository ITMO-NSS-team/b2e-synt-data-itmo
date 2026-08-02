"""Telegram bot — a pass-through surface for researchers.

Deliberately thin. The spec is explicit: "no additional logics in it – just
by-passing to B2E". So this holds exactly one piece of state, the mapping from a
Telegram chat to a B2E session, and otherwise forwards text in and text out.

That thinness is a measurement property, not laziness. Any retry, any rewording,
any "helpful" summarisation here would appear in the traces as agent behaviour
and quietly contaminate every comparison a researcher makes between manual and
batch runs.

A chat is a conversation
------------------------
The chat -> session mapping outlives a single question, and the session it points
at is opened against ``agent_config_interactive`` — the shipped config whose
``conversation_mode`` is ``resume``. Consecutive messages therefore land in one
headless Claude Code session, and the agent sees its own earlier tool calls and
results rather than meeting each question cold.

That is what makes a clarifying question worth asking here. The CLI has no
``AskUserQuestion`` in headless mode (probed on 2.1.220 — see
``sim/agent/claude_code.py``), so the turn boundary *is* the asking mechanism:
the agent ends a turn with a question, the researcher answers in the next
message, and the session continues with everything still in context.

``/start_new_session`` drops the mapping. The next message opens a fresh session
with no history — which is the thing you want when the previous thread has
wandered, or when you are about to measure something and do not want the last
ten messages priced into the context of the first.

Showing that something is happening
-----------------------------------
A turn runs upwards of a minute. Silence for that long reads as a dead bot, so
the bridge posts one status message and rewrites it in place from
``GET /sessions/{id}/progress`` — "запрашиваю данные → изучаю структуру
витрины… 34 с, шагов: 6".

This is the one place the bridge does more than forward bytes, so the limits are
worth stating. The status is a separate message: the question and the answer are
still passed through verbatim. The progress endpoint is read-only and never
reaches the model, so nothing here shows up in a trace as agent behaviour. And
every failure in the status path — a 404, a dropped poll, a refused edit — is
swallowed, because a broken progress display must never cost a turn that the
agent is otherwise completing and being paid for.

Long polling rather than webhooks: a webhook needs a public HTTPS endpoint for
Telegram to call, which would mean another hole in the reverse proxy. Polling
keeps the bot strictly outbound, so it adds no exposed surface at all.

No external dependency — the Bot API is plain HTTPS/JSON, and adding a library
for three endpoints would be a poor trade on this box.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

log = logging.getLogger("b2e.telegram")

API = "https://api.telegram.org/bot{token}/{method}"

HELP = (
    "B2E — исследовательский стенд.\n\n"
    "Просто напишите вопрос: он уйдёт в агента как есть.\n"
    "Сообщения идут одной сессией — агент помнит предыдущие ходы, "
    "так что уточнения и «а теперь то же самое, но по отделу» работают.\n\n"
    "Ход занимает от полуминуты до нескольких минут. Пока он идёт, "
    "сообщение со статусом обновляется на месте — видно, какие витрины "
    "агент сейчас читает.\n\n"
    "/start_new_session — начать сессию заново, без прежнего контекста\n"
    "/whoami — личность, сессия и конфигурация\n"
    "/employee <id> — переключить личность\n"
    "/help — эта справка\n\n"
    "Это стенд, а не продукт: агент может ошибаться, и проверять его — "
    "часть работы. Номер trace под каждым ответом ведёт в полную запись хода."
)

#: ``/new`` predates ``/start_new_session`` and did the same thing. Kept, because
#: removing a command someone has in their muscle memory is a silent failure:
#: Telegram does not error on an unknown command, it just forwards the text to
#: the agent, and the researcher gets an answer to "/new" instead of a reset.
NEW_SESSION_COMMANDS = ("/start_new_session", "/new")

#: Registered with Telegram at startup so the client shows a "/" menu. Without
#: this the commands exist but are invisible — discoverable only by reading
#: /help, which nobody does twice.
#:
#: `/new` is deliberately absent: it still works, but listing two names for one
#: action in a menu invites the reader to look for a difference that is not
#: there. Descriptions are capped at 256 characters by the Bot API.
BOT_COMMANDS = [
    {"command": "start_new_session",
     "description": "Начать заново — сбросить контекст разговора"},
    {"command": "whoami",
     "description": "Личность, сессия и конфигурация"},
    {"command": "employee",
     "description": "Сменить личность: /employee <id>"},
    {"command": "help",
     "description": "Что это и как пользоваться"},
]


class TelegramBridge:
    """Chat id -> B2E session. Nothing else is remembered here."""

    def __init__(self, *, token: str, agent_url: str, default_employee: str,
                 config_ref: str = "agent_config_interactive",
                 poll_seconds: float = 4.0) -> None:
        if not token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is unset. It is read from the environment "
                "only and is never written to a file or a commit."
            )
        self.token = token
        self.agent_url = agent_url.rstrip("/")
        self.default_employee = default_employee
        self.config_ref = config_ref
        # Slow enough that a long turn costs a handful of edits rather than
        # dozens — Telegram rate-limits edits per chat, and a status that
        # flickers is not more informative than one that does not.
        self.poll_seconds = poll_seconds
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

    def send(self, chat_id: int, text: str) -> int | None:
        """Send text, returning the id of the last message sent.

        Telegram rejects messages over 4096 characters. Chunking is transport,
        not logic: the text is not altered, only split.
        """
        message_id = None
        for i in range(0, len(text) or 1, 4000):
            reply = self._call("sendMessage", chat_id=chat_id,
                               text=text[i:i + 4000] or "(пустой ответ)")
            message_id = (reply.get("result") or {}).get("message_id", message_id)
        return message_id

    def edit(self, chat_id: int, message_id: int | None, text: str) -> None:
        """Rewrite a message in place, best effort.

        Failures are swallowed on purpose. This only ever carries the status
        line, so a failed edit costs a stale progress display — while raising
        would abandon a turn the agent is still paying for. Telegram also
        rejects an edit whose text is unchanged, which is not an error worth
        hearing about.
        """
        if message_id is None:
            return
        try:
            self._call("editMessageText", chat_id=chat_id,
                       message_id=message_id, text=text[:4000])
        except httpx.HTTPError as exc:
            log.debug("status edit failed: %s", exc)

    # --------------------------------------------------------------- agent

    def ensure_session(self, chat_id: int) -> str:
        if chat_id in self.sessions:
            return self.sessions[chat_id]
        employee = self.employees.get(chat_id, self.default_employee)
        response = self._agent.post(
            f"{self.agent_url}/sessions",
            json={"employee_id": employee, "config_ref": self.config_ref})
        response.raise_for_status()
        session_id = response.json()["session_id"]
        self.sessions[chat_id] = session_id
        return session_id

    def ask(self, chat_id: int, text: str) -> str:
        """Ask the agent, keeping a status line updated while it works.

        A turn runs upwards of a minute — 89 s on the deployed stack — and until
        now that was a minute of nothing, which reads as a broken bot. The
        question and the answer still pass through untouched; the only addition
        is a separate message that the bridge rewrites in place with whatever
        `GET /sessions/{id}/progress` reports.

        The turn runs on its own thread purely so this one can poll. The poll is
        read-only and never reaches the model, so nothing here appears in a
        trace as agent behaviour — the thinness that makes manual and batch runs
        comparable is preserved.
        """
        session_id = self.ensure_session(chat_id)
        status_id = self.send(chat_id, "⏳ принял вопрос, начинаю…")

        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                outcome["response"] = self._agent.post(
                    f"{self.agent_url}/sessions/{session_id}/messages",
                    json={"content": text})
            except Exception as exc:                      # noqa: BLE001
                outcome["error"] = exc

        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        shown = ""
        while True:
            worker.join(self.poll_seconds)
            if not worker.is_alive():
                break
            line = self.progress_line(session_id)
            if line and line != shown:
                shown = line
                self.edit(chat_id, status_id, f"⏳ {line}")

        if "error" in outcome:
            self.edit(chat_id, status_id, "⚠️ не дошло до агента")
            raise outcome["error"]

        response = outcome["response"]
        if response.status_code >= 400:
            self.edit(chat_id, status_id, f"⚠️ агент ответил {response.status_code}")
            return f"[b2e {response.status_code}] {response.text[:600]}"

        body = response.json()
        stats = body.get("stats", {})
        self.edit(chat_id, status_id, "✅ готово")
        # The footer is diagnostics for a researcher, appended after the answer.
        # It never changes the question or the answer.
        footer = (f"\n\n— {stats.get('heimdall_calls', 0)} вызовов API, "
                  f"{stats.get('total_tokens', 0)} токенов, "
                  f"trace {str(body.get('trace_id'))[:16]}")
        return (body.get("answer") or "(пустой ответ)") + footer

    def progress_line(self, session_id: str) -> str:
        """Current status of the in-flight turn, or "" if there is nothing.

        Every failure is a silent "": a 404 means the turn has not registered
        yet or has already been evicted, and anything else on a *status* poll
        must not disturb a turn that is otherwise going fine.

        The catch is deliberately bare. Narrowing it to httpx errors is the
        instinct, and it is wrong here: this path is decoration on top of a turn
        that costs real money and takes real minutes, so *any* exception it
        raises would trade an answer for a cosmetic detail. Caught by a test
        that asserts exactly that.
        """
        try:
            response = self._agent.get(
                f"{self.agent_url}/sessions/{session_id}/progress", timeout=10.0)
            if response.status_code != 200:
                return ""
            return str(response.json().get("status") or "")
        except Exception as exc:                          # noqa: BLE001
            log.debug("progress poll failed: %s", exc)
            return ""

    # ---------------------------------------------------------------- loop

    def handle(self, message: dict[str, Any]) -> None:
        chat_id = message["chat"]["id"]
        text = (message.get("text") or "").strip()
        if not text:
            return

        # The whole first word, not a prefix. `startswith("/new")` also matched
        # `/newcomers`, so a question about new hires reset the session and was
        # never asked — and `startswith("/start")` would now swallow
        # `/start_new_session`. The `@botname` suffix is what Telegram appends
        # to every command in a group chat.
        command = text.split(maxsplit=1)[0].split("@")[0]

        if command in ("/start", "/help"):
            self.send(chat_id, HELP)
            return
        if command in NEW_SESSION_COMMANDS:
            previous = self.sessions.pop(chat_id, None)
            self.send(chat_id, (
                f"Новая сессия. Прежний контекст сброшен (была {previous}); "
                f"сама сессия откроется на первом же вопросе."
                if previous else
                "Открытой сессии и не было — следующий вопрос начнёт новую."))
            return
        if command == "/whoami":
            employee = self.employees.get(chat_id, self.default_employee)
            self.send(chat_id, f"employee_id = {employee}\n"
                               f"session = {self.sessions.get(chat_id, '—')}\n"
                               f"config_ref = {self.config_ref}")
            return
        if command == "/employee":
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
            # Deliberately not retried. A resend would double-charge a turn that
            # may well have completed on the agent's side, and it would appear
            # in the traces as agent behaviour — which is exactly the
            # contamination this bridge stays thin to avoid. So: say plainly
            # what is known and let the researcher decide.
            #
            # The commonest cause by far is the agent container restarting
            # mid-turn, which is what a deploy looks like from here.
            self.send(chat_id,
                      f"⚠️ связь с агентом оборвалась: {exc}\n\n"
                      f"Вопрос мог быть выполнен, а мог и нет — я не знаю и не "
                      f"повторяю его сам, чтобы не оплатить ход дважды. "
                      f"Обычно это перезапуск агента. Повторите вопрос.")

    def register_commands(self) -> None:
        """Publish the command menu. Best effort, once, at startup.

        A failure here costs discoverability, not function — every command still
        works when typed — so it must not stop the bridge from starting. The
        call is idempotent: Telegram stores the list against the bot, so
        re-registering an unchanged list is a no-op.
        """
        try:
            reply = self._call("setMyCommands", commands=BOT_COMMANDS)
            if not reply.get("ok"):
                log.warning("setMyCommands refused: %s", reply)
        except httpx.HTTPError as exc:
            log.warning("could not register the command menu: %s", exc)

    def run(self) -> None:
        self.register_commands()
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
    # httpx logs the full request URL at INFO, and every Bot API URL contains
    # the bot token — so an hour of polling writes the credential into the
    # container log a hundred times, where `docker logs` hands it to anyone who
    # can read it. Warnings still come through; only the per-request line goes.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    TelegramBridge(
        token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        agent_url=os.environ.get("B2E_AGENT_URL", "http://b2e-agent:8082"),
        default_employee=os.environ.get("TELEGRAM_DEFAULT_EMPLOYEE", ""),
        config_ref=os.environ.get("B2E_TELEGRAM_CONFIG_REF",
                                  "agent_config_interactive"),
    ).run()


if __name__ == "__main__":
    main()
