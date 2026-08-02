"""Live turn progress, for surfaces where somebody is waiting.

A turn through the Claude Code harness takes upwards of a minute — measured on
the deployed stack: 89 s, 13 iterations, 7 Heimdall calls. Over a Telegram
bridge that is a minute of silence, which is indistinguishable from the bot
being broken. This module is what makes the wait legible.

What it is not
--------------
It is not a second source of truth. The trace is the record; this is a view of
one in-flight turn that is discarded the moment the turn ends. Nothing here is
persisted, nothing here is fingerprinted, and no consumer of it may influence
the turn — the Telegram bridge polls it read-only and the poll never reaches the
model. Keeping that line sharp is what stops a progress display from quietly
becoming agent behaviour that shows up in somebody's measurements.

Why the mapping lives here rather than in the bot
-------------------------------------------------
Because the bot is supposed to stay thin. Translating `mcp__heimdall__mcp_query`
into "запрос к витрине" is knowledge about the harness, and the harness is here.
The bot renders strings it is given.
"""
from __future__ import annotations

import threading
import time
from typing import Any

MCP_PREFIX = "mcp__heimdall__"

#: Tool name -> what to tell a human waiting on it. Keyed by the *granted* tool
#: names, so a tool that is added to the surface without a line here degrades to
#: its own name rather than vanishing from the display.
TOOL_LABELS = {
    "get_overview": "смотрю, что вообще есть в Heimdall",
    "get_docs": "читаю документацию витрин",
    "find_skills": "ищу подходящий скилл",
    "get_skill": "разбираю скилл",
    "list_models": "перебираю витрины",
    "describe_model": "изучаю структуру витрины",
    "mcp_query": "запрашиваю данные",
    "ToolSearch": "подгружаю инструменты",
    "Bash": "запускаю одобренный скилл",
}

#: How many recent steps the rendered line keeps. A turn can make a dozen calls
#: and a status message that grows without bound is worse than no status at all.
TRAIL = 3


def label_for(tool_name: str) -> str:
    short = str(tool_name or "").removeprefix(MCP_PREFIX)
    return TOOL_LABELS.get(short, short or "работаю")


class TurnProgress:
    """Mutable state of one in-flight turn. Thread-safe by a single lock.

    Written from the thread draining the CLI's stdout and read from whichever
    request thread is polling, so every field goes through the lock — a dict
    read mid-update would otherwise hand a poller a step count from one moment
    and a tool name from another.
    """

    def __init__(self, session_id: str, question: str) -> None:
        self._lock = threading.Lock()
        self.session_id = session_id
        self.question = question
        self.started_at = time.time()
        self.updated_at = self.started_at
        self.steps: list[str] = []
        self.tool_calls = 0
        self.heimdall_calls = 0
        self.done = False
        self.failed = False

    # ------------------------------------------------------------- writing

    def observe(self, event: dict[str, Any]) -> None:
        """Fold one stream-json event into the current state."""
        kind = event.get("type")
        if kind == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    self._step(label_for(block.get("name")),
                               heimdall=str(block.get("name") or "")
                               .startswith(MCP_PREFIX))
        elif kind == "result":
            with self._lock:
                self.done = True
                self.failed = bool(event.get("is_error"))
                self.updated_at = time.time()

    def _step(self, label: str, *, heimdall: bool) -> None:
        with self._lock:
            self.tool_calls += 1
            if heimdall:
                self.heimdall_calls += 1
            # Consecutive identical steps are collapsed. Paging a mart is five
            # `mcp_query` calls in a row, and five identical lines reads as a
            # stutter rather than as progress.
            if not self.steps or self.steps[-1] != label:
                self.steps.append(label)
            self.updated_at = time.time()

    def finish(self, *, failed: bool = False) -> None:
        with self._lock:
            self.done = True
            self.failed = self.failed or failed
            self.updated_at = time.time()

    # ------------------------------------------------------------- reading

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "elapsed_seconds": round(time.time() - self.started_at, 1),
                "steps": list(self.steps),
                "tool_calls": self.tool_calls,
                "heimdall_calls": self.heimdall_calls,
                "done": self.done,
                "failed": self.failed,
                "status": self._render(),
            }

    def _render(self) -> str:
        """One line a human can read. Called under the lock."""
        elapsed = int(time.time() - self.started_at)
        if self.done:
            return (f"готово за {elapsed} с, "
                    f"{self.heimdall_calls} обращений к Heimdall")
        if not self.steps:
            return f"думаю… {elapsed} с"
        trail = " → ".join(self.steps[-TRAIL:])
        return f"{trail}… {elapsed} с, шагов: {self.tool_calls}"


class ProgressBoard:
    """Every in-flight turn, keyed by session.

    In memory and deliberately not in the store: it describes a process, not a
    result, and a progress row that outlived its process would be a lie the next
    reader has no way to detect. One entry per session — a session runs one turn
    at a time, and a second concurrent question simply replaces the view.
    """

    def __init__(self, retain_seconds: float = 120.0) -> None:
        self._lock = threading.Lock()
        self._turns: dict[str, TurnProgress] = {}
        self.retain_seconds = retain_seconds

    def start(self, session_id: str, question: str) -> TurnProgress:
        turn = TurnProgress(session_id, question)
        with self._lock:
            self._evict()
            self._turns[session_id] = turn
        return turn

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            turn = self._turns.get(session_id)
        return turn.snapshot() if turn is not None else None

    def _evict(self) -> None:
        """Drop finished turns that nobody came back for. Called under the lock.

        Kept briefly after completion on purpose: a poller that asks one last
        time should see "done" rather than a 404, which is indistinguishable
        from a session that never ran.
        """
        cutoff = time.time() - self.retain_seconds
        for key, turn in list(self._turns.items()):
            if turn.done and turn.updated_at < cutoff:
                del self._turns[key]
