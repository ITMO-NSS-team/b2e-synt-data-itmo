"""The one LLM call in reflection: propose lesson *text*, never evidence.

Why the output schema has no ``support``/``refute``
-----------------------------------------------------
``sim.reflection.memory``'s module docstring names the failure this closes: a
curator that both discovers a pattern and counts how strong it is will round
its own hunch up, and two episodes becomes "this fails often". So the three
op types below — ``AddOp``, ``ReviseOp``, ``DropOp`` — simply have no field a
count could occupy. ``parse_ops`` reads a curator response into these types by
picking named keys off a dict; an ``ADD`` entry that also carries
``"support": 99`` is not rejected, because rejecting it would be giving the
model's stray field the power to fail the whole response — it is silently
never read, which is a stronger guarantee than validation could give: there is
no code path in this module through which a count reaches an ``Op``.

Why transport is ``claude -p`` and not ``sim.agent.llm``
-----------------------------------------------------------
``sim/agent/llm.py``'s ``AnthropicClient`` talks to ``/v1/messages`` directly,
and ``docs/assumptions.md`` A-8 records that this deployment's credential
(``CLAUDE_CODE_OAUTH_TOKEN`` — ``ANTHROPIC_API_KEY`` is empty here) is refused
there with a 403 on every header form tried. The sanctioned use of that
credential is the ``claude`` CLI itself, so ``ClaudeCLICuratorClient`` below
drives it exactly the way ``sim.agent.claude_code.ClaudeCodeHarness`` does for
the agent under study — same ``child_env()`` credential lookup, same
``--setting-sources ""`` isolation from the host's own Claude config — except
the curator gets **every** tool disallowed. It has no legitimate reason to read
a file or call Heimdall: its whole input is already assembled in the prompt,
and a curator that could query the live API could look answers up, which is
the one channel the rest of Plan B works hard to close.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence

from sim.agent.claude_code import CODE_EXECUTION_TOOLS, DENIED_TOOLS, PERMISSION_MODE, \
    ClaudeCodeHarness
from sim.agent.config import DEFAULT_MODEL
from sim.reflection.memory import MEMORY_KINDS
from sim.registry import Registry, sha256_hex

#: At most this many operations per call. The number itself is arbitrary
#: (the spec states it without deriving it), but bounding it at all matters:
#: an unbounded op list is an unbounded, ungated amount of agent-facing text
#: added to memory from a single model response.
MAX_OPS = 8

#: Every built-in tool the curator CLI session could reach, disallowed without
#: exception. Reflection reads spans that are already extracted into cards and
#: a table; it has no legitimate reason to open a file, run a shell command, or
#: reach the network, and a curator that could query the live Heimdall API
#: could simply look up the answer it is supposed to be teaching method around.
CURATOR_DISALLOWED_TOOLS = tuple(sorted(set(DENIED_TOOLS) | set(CODE_EXECUTION_TOOLS)))


# ------------------------------------------------------------------------ ops


@dataclass(frozen=True, slots=True)
class AddOp:
    """Propose a brand-new lesson. No ``support``/``refute`` field exists on
    this type — see the module docstring for why that omission is the point."""

    kind: str
    trigger: dict[str, Any]
    text: str

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"kind {self.kind!r} not in {MEMORY_KINDS}")


@dataclass(frozen=True, slots=True)
class ReviseOp:
    """Replace an existing item's wording. Never its trigger or kind — those
    are what its accumulated ``support``/``refute`` were measured against, and
    letting a rewording also move the trigger would let the model quietly
    redirect evidence collected under one situation onto another."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class DropOp:
    """Retire an item. ``reason`` is a curator-log field only — it is never
    rendered to the agent, so it does not pass through ``guard.check``."""

    id: str
    reason: str


Op = AddOp | ReviseOp | DropOp


def _extract_json_text(raw: str) -> str:
    """Best-effort recovery of a JSON array from a chat-style model response.

    A ``claude -p`` text response routinely wraps the answer in a code fence
    or a sentence ("Вот список операций:\\n```json\\n[...]\\n```"), and a
    strict ``json.loads`` on the raw string would fail on the majority of
    real responses rather than the minority. This looks for the first ``[``
    and the last ``]`` in the text and hands that slice to the caller; if no
    bracket pair exists the original string is returned unchanged so the
    caller's own ``json.loads`` failure explains what went wrong.
    """
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def parse_ops(raw: str) -> list[Op]:
    """The curator's response, reduced to at most ``MAX_OPS`` typed ops.

    Malformed individual entries are skipped rather than failing the whole
    batch: a curator call producing seven good ops and one garbled one should
    not lose the seven, and there is no retry here to give a second chance to
    (see ``sim.reflection.guard`` for why retrying is refused on principle,
    not just on this path). A response that is not JSON at all, or contains
    no array, yields an empty list — silence, not a crash, because "the model
    said nothing usable this epoch" is a normal outcome (see
    ``sim.reflection.reflect``'s "a night that learns nothing" test).
    """
    try:
        parsed = json.loads(_extract_json_text(raw))
    except (ValueError, TypeError):
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("ops", [])
    if not isinstance(parsed, list):
        return []

    ops: list[Op] = []
    for entry in parsed:
        if len(ops) >= MAX_OPS:
            break
        if not isinstance(entry, dict):
            continue
        kind_of_op = str(entry.get("op") or "").upper()
        try:
            if kind_of_op == "ADD":
                trigger = entry.get("trigger")
                ops.append(AddOp(
                    kind=str(entry["kind"]),
                    trigger=dict(trigger) if isinstance(trigger, dict) else {},
                    text=str(entry["text"]),
                ))
            elif kind_of_op == "REVISE":
                ops.append(ReviseOp(id=str(entry["id"]), text=str(entry["text"])))
            elif kind_of_op == "DROP":
                ops.append(DropOp(id=str(entry["id"]),
                                  reason=str(entry.get("reason") or "")))
            # Any other/missing "op" value: not one of the three ops this
            # module knows how to apply, so it is skipped rather than guessed
            # at — the same "no code path reads a field we did not name"
            # discipline the module docstring describes for `support`.
        except (KeyError, ValueError):
            continue
    return ops


# --------------------------------------------------------------------- prompt


def _op_schema_note() -> str:
    return (
        "Формат ответа — JSON-массив, не более 8 элементов, каждый элемент "
        "один из трёх видов:\n"
        '  {"op": "ADD", "kind": "api_mechanic|method|pitfall", '
        '"trigger": {...}, "text": "..."}\n'
        '  {"op": "REVISE", "id": "...", "text": "..."}\n'
        '  {"op": "DROP", "id": "...", "reason": "..."}\n'
        "Никаких полей support и refute — их считает не модель, а код, "
        "который получит этот ответ. Текст — по-русски, короткий "
        "(меньше 240 символов, не больше двух предложений), про то, как "
        "работать с API, а не про то, каким должен быть ответ на вопрос."
    )


def build_prompt(cards: Sequence[Any], table: dict[str, Any],
                 memory: Sequence[Any]) -> str:
    """The one prompt the curator ever sees, assembled from already-computed,
    value-free material.

    ``table["_fix_diffs"]`` is dropped before serialising: it is internal
    plumbing ``aggregate()`` keeps for ``build_episode_card`` (see that
    module's docstring), not part of the aggregation table a curator prompt
    should render, and its keys are tuples that ``json.dumps`` cannot encode
    as-is.
    """
    visible_table = {k: v for k, v in table.items() if not k.startswith("_")}
    cards_payload = [
        {"question_class": c.question_class, "family": c.family,
         "category": c.category, "plan": list(c.plan),
         "api_errors": list(c.api_errors), "verdict": c.verdict,
         "failure_class": c.failure_class, "calls": c.calls,
         "tokens": c.tokens, "seconds": c.seconds}
        for c in cards
    ]
    memory_payload = [
        {"id": m.id, "kind": m.kind, "trigger": m.trigger, "text": m.text,
         "support": m.support, "refute": m.refute}
        for m in memory
    ]
    return (
        "Ты — куратор долговременной памяти агента, который отвечает на "
        "вопросы через API Heimdall. Ниже — сводка за одну эпоху: карточки "
        "эпизодов (без ответов и без эталонных значений), агрегированная "
        "таблица счётчиков и текущая память.\n\n"
        f"Карточки эпизодов:\n{json.dumps(cards_payload, ensure_ascii=False)}\n\n"
        f"Агрегированная таблица:\n{json.dumps(visible_table, ensure_ascii=False)}\n\n"
        f"Текущая память:\n{json.dumps(memory_payload, ensure_ascii=False)}\n\n"
        "Предложи операции ADD/REVISE/DROP, которые улучшат память для "
        "следующей эпохи. Не упоминай конкретных людей, подразделения, "
        "числа за пределами таблицы выше или формулировки вопросов из "
        "корзины. " + _op_schema_note()
    )


# --------------------------------------------------------------------- client


class CuratorClient(Protocol):
    """Anything that can turn a prompt into the curator's raw text response.

    Deliberately this narrow — ``complete(prompt) -> str`` — so a test's fake
    client and the real CLI-backed one are interchangeable with no adapter,
    which is exactly what ``test_shared_and_isolated_differ_only_in_input``
    (``tests/test_reflection_reflect.py``) relies on to hold the model call
    itself constant while only the episodes vary.
    """

    def complete(self, prompt: str) -> str: ...


@dataclass(slots=True)
class ClaudeCLICuratorClient:
    """Runs the curator's call through ``claude -p --output-format json``.

    Deliberately not a ``ClaudeCodeHarness`` subclass or a new copy of its
    argv/env logic: an actual harness instance is built (with unused Heimdall
    fields, since the curator never touches Heimdall) purely so
    ``child_env()`` — the credential lookup ``docs/assumptions.md`` A-8 says
    must be an OAuth token, not an API key — is read exactly once, from the
    one place that already reads it correctly.
    """

    model_id: str = DEFAULT_MODEL
    claude_bin: str = "claude"
    claude_home: str | None = None
    proxy: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 180

    def complete(self, prompt: str) -> str:
        harness = ClaudeCodeHarness(
            heimdall_url="", heimdall_token="",
            claude_bin=self.claude_bin, claude_home=self.claude_home,
            proxy=self.proxy,
        )
        argv = [
            harness.claude_bin, "-p", prompt,
            "--model", self.model_id,
            "--output-format", "json",
            "--setting-sources", "",
            "--permission-mode", PERMISSION_MODE,
            "--disallowed-tools", ",".join(CURATOR_DISALLOWED_TOOLS),
        ]
        proc = subprocess.run(
            argv, env=harness.child_env(), capture_output=True, text=True,
            timeout=self.timeout_seconds,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"curator CLI call failed (exit {proc.returncode}): "
                f"{proc.stderr[-2000:]}")
        try:
            envelope = json.loads(proc.stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"curator CLI produced non-JSON output: {proc.stdout[:2000]!r}"
            ) from exc
        return str(envelope.get("result", ""))


@dataclass(slots=True)
class CachingCuratorClient:
    """Wraps any ``CuratorClient`` with content-hash caching in the registry.

    Re-deriving an epoch — replaying the same cards, table and memory through
    the same prompt — then costs a registry read, not a subprocess spawn, and
    a test can seed the registry with a cached response and never touch a
    live model at all. Keyed on the *prompt* text: two calls whose assembled
    prompts are byte-identical are, by construction, the same question asked
    of the model, regardless of which episodes or scope produced that prompt.
    """

    inner: "CuratorClient"
    registry: Registry
    actor: str = "reflection"

    def complete(self, prompt: str) -> str:
        key = "curator_cache_" + sha256_hex(prompt.encode("utf-8"))
        try:
            _version, cached = self.registry.load(key)
            return str(cached["response"])
        except KeyError:
            pass
        response = self.inner.complete(prompt)
        self.registry.commit(key, "curator_cache", {"response": response},
                             actor=self.actor, note="curator response cache")
        return response


# --------------------------------------------------------------------- curate


def curate(cards: Sequence[Any], table: dict[str, Any], memory: Sequence[Any],
          *, client: "CuratorClient") -> list[Op]:
    """Step 4 of reflection: one prompt, one response, at most 8 typed ops.

    Everything the model is allowed to see is already value-free
    (``cards``/``table`` from ``sim.reflection.aggregate``) or evidence-free
    on the way out (``parse_ops`` drops any counters). This function does not
    call ``sim.reflection.guard`` — that happens in
    ``sim.reflection.reflect`` after ``curate`` returns, because the guard's
    per-rule rejection counts are reflection's own bookkeeping, not something
    a curation call should know about (a curator that could see which rule
    rejected its last attempt would start optimising against the rule instead
    of against writing a good lesson — the same "no retry" reasoning
    ``guard.py`` states for itself).
    """
    prompt = build_prompt(cards, table, memory)
    raw = client.complete(prompt)
    return parse_ops(raw)


# ---------------------------------------------------------------- shared merge


def _miner_key(item: Any) -> str:
    """The exact-key identity of a lesson: same kind, same trigger, same
    wording. Deliberately exact rather than fuzzy — see
    ``sim.reflection.reflect``'s ``mint_id`` for why, and the design spec's
    ``merge_shared`` table for the corpus property (paraphrases written to be
    lexically distant from their bases, caution questions near-identical to
    their answerable siblings) that makes any embedding-based merge actively
    wrong here rather than merely imprecise."""
    return item.id


#: See ``merge_shared``'s docstring. A flat module constant, not a magic
#: number inside the function body, so a reviewer can see and cite the exact
#: size of the adjustment.
_CROSS_INSTANCE_BONUS = 1


def merge_shared(per_instance_candidates: Sequence[Any]) -> list[Any]:
    """Combine per-instance ``MemoryItem`` candidates into one shared list.

    Grouped by ``id`` — which is already a content hash of
    ``(kind, trigger, text)`` (see ``sim.reflection.reflect.mint_id``), so two
    instances proposing the *same* lesson collide on it with no coordination,
    and two instances proposing *differently-worded* lessons for the same
    trigger do not collide at all, which is intentional: merging those would
    be exactly the similarity-based merge the design spec forbids.

    Episode ids are **unioned as sets, never summed**: an episode counted by
    two instances is still one episode, and ``support``/``refute`` are
    recomputed from the size of the union rather than carried over from the
    inputs, so a bug that let the same episode reach two candidates cannot
    silently double that episode's weight — recomputing from the ground-truth
    id sets cannot drift the way adding two pre-computed counts could.

    Items whose merged evidence spans two or more distinct instances get a
    small explicit corroboration bonus on top of their union-derived support.
    This is not the model writing its own evidence (the bonus is applied here,
    in code, after curation, to already-committed episode ids) — it is this
    function's own, documented adjustment for a real statistical difference:
    the same conclusion reached independently by two instances is stronger
    evidence than the same total count observed by one.
    """
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in per_instance_candidates:
        groups[_miner_key(item)].append(item)

    merged: list[Any] = []
    for group in groups.values():
        episodes_support: set[str] = set()
        episodes_refute: set[str] = set()
        origin_instances: set[str] = set()
        for it in group:
            episodes_support |= set(it.episodes_support)
            episodes_refute |= set(it.episodes_refute)
            origin_instances |= set(it.origin_instances)

        support = len(episodes_support)
        refute = len(episodes_refute)
        if len(origin_instances) >= 2:
            support += _CROSS_INSTANCE_BONUS

        base = group[0]
        merged.append(replace(
            base,
            support=support, refute=refute,
            episodes_support=tuple(sorted(episodes_support)),
            episodes_refute=tuple(sorted(episodes_refute)),
            origin_instances=tuple(sorted(origin_instances)),
            created_epoch=min(it.created_epoch for it in group),
            last_useful_epoch=max(it.last_useful_epoch for it in group),
            epochs_idle=min(it.epochs_idle for it in group),
        ))
    return merged
