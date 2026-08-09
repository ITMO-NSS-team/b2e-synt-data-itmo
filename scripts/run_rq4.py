#!/usr/bin/env python
"""The RQ4 experiment driver: three arms, nine epochs, one seeded schedule.

Shape of a run
--------------
For each replication, for each epoch 0..8: every arm (A1, A2, A3) answers the
same 30 :class:`sim.oracle.schedule.ScheduledItem` questions, 10 per instance,
via ``POST /sessions`` then ``POST /sessions/{id}/messages`` against the b2e
agent service. Each answer is scored immediately and appended as one line to
``var/rq4/<run_id>/turns.jsonl`` — see ``TURNS_JSONL_SCHEMA`` below for the
exact row shape Task 3 reads. Once every arm's turns for an epoch are done,
A2 runs reflection three times (once per instance, over that instance's own
10 episodes) and A3 runs it once (over all 30 pooled episodes, with
``sim.reflection.curate.merge_shared`` folded in — see ``_reflect_a3_pooled``
for why that is not simply "call ``reflect()`` with more episodes"), and each
arm/instance gets a new ``agent_config_rq4_<arm>_i<instance>`` version pinning
whatever memory now exists, read by the next epoch's sessions.

Where this runs
----------------
The agent service (``b2e-sim-b2e-agent-1``) listens on :8082 *inside* the
compose network only, and its registry mount is read-only (the threat model
forbids the agent uid from writing the approval store). Reflection's curator
call also needs the ``claude`` CLI, which only exists in the ``agent`` Docker
image target (the emulator/admin-ui/research-api image has no Node runtime).
So this script is meant to run inside a *throwaway* container built from that
same image, with the repo bind-mounted (no rebuild needed for a code change)
and the registry volume mounted **read-write** — overriding the b2e-agent
service's own restricted mount, which is a property of that service's compose
entry, not of the image or the volume. Verified live on 2026-08-09 (see the
Task 1/2 report):

    docker run --rm -it \\
      --network b2e-sim_edge --network b2e-sim_internal \\
      -v "$(pwd)":/app -v b2e-sim_registry_data:/app/registry \\
      --env-file deploy/.env \\
      -w /app b2e-sim/agent:local \\
      python scripts/run_rq4.py --seed 20260809 --hr-employee-id 9877478 ...

Both networks matter: ``internal`` reaches ``b2e-agent``/``phoenix`` by service
name, ``edge`` reaches the proxy-relay that puts the curator's ``claude -p``
calls on the path to Anthropic (``AGENT_HTTP_PROXY`` in ``deploy/.env``) — a
container on ``internal`` alone cannot reach the relay's bind address, because
compose gives every network its own bridge and Docker does not route between
them by default.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.agent.config import AgentConfig                                   # noqa: E402
from sim.oracle.labels import GoldLabels                                    # noqa: E402
from sim.oracle.reference import evaluate as reference_evaluate             # noqa: E402
from sim.oracle.schedule import (N_EPOCHS, N_INSTANCES, ScheduledItem,      # noqa: E402
                                 build_schedule, commit_design)
from sim.reflection import curate as curate_mod                             # noqa: E402
from sim.reflection import reflect as reflect_mod                           # noqa: E402
from sim.reflection.aggregate import Episode, aggregate, build_episode_card # noqa: E402
from sim.reflection.extract import call_records, turn_facts                 # noqa: E402
from sim.reflection.guard import RULE_IDS                                   # noqa: E402
from sim.reflection.memory import (MemoryItem, build_pack,                  # noqa: E402
                                   commit_pack, load_pack)
from sim.research.answer import parse_answer                                # noqa: E402
from sim.research.evaluate import TraceFacts, score_correctness             # noqa: E402
from sim.research.phoenix_client import PhoenixClient                       # noqa: E402
from sim.registry import Registry                                           # noqa: E402

ARMS: tuple[str, ...] = ("A1", "A2", "A3")

#: The exact row shape written to turns.jsonl, one line per attempted turn
#: (dispatched successfully or not). Task 3's report.py reads this format;
#: keep the two in sync deliberately rather than by convention.
#:
#: replication:int, arm:str, epoch:int, instance:int, question_id:str,
#: pair_id:str, category:str, family:str, question_class:str,
#: employee_id:str, config_ref:str, dispatched_at:float(unix), attempts:int,
#: dispatch_failed:bool, dispatch_reason:str|null,
#: session_id:str|null, trace_id:str|null, wall_seconds:float|null,
#: scored:bool, correct:bool|null, score_reason:str,
#: answer_present:bool, answer_verdict:str|null, answer_ids:list[str],
#: answer_value:float|null, answer_refused:bool, answer_reason:str|null,
#: answer_field_errors:dict, tokens:int, seconds:float, heimdall_calls:int,
#: memory_tokens:int, dry_run:bool
TURNS_JSONL_SCHEMA = (
    "replication", "arm", "epoch", "instance", "question_id", "pair_id",
    "category", "family", "question_class", "employee_id", "config_ref",
    "dispatched_at", "attempts", "dispatch_failed", "dispatch_reason",
    "session_id", "trace_id", "wall_seconds", "scored", "correct",
    "score_reason", "answer_present", "answer_verdict", "answer_ids",
    "answer_value", "answer_refused", "answer_reason", "answer_field_errors",
    "tokens", "seconds", "heimdall_calls", "memory_tokens", "dry_run",
)


# --------------------------------------------------------------------- errors


class DispatchFailed(Exception):
    """A turn could not be dispatched after every retry. Recorded as a row,
    never raised out of the driver — an aborted arm is unrecoverable; a row
    with ``dispatch_failed=True`` is just an unscored row the report already
    knows how to handle."""

    def __init__(self, reason: str, attempts: int) -> None:
        self.reason = reason
        self.attempts = attempts
        super().__init__(reason)


# ------------------------------------------------------------------ HTTP client


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


@dataclass
class AgentClient:
    """Talks to ``POST /sessions`` and ``POST /sessions/{id}/messages``, with
    exponential backoff and jitter on 429/5xx/transport failure.

    ``sleep`` and ``rand`` are injected so a test can run the retry loop
    without a real delay and without depending on wall-clock jitter for its
    assertions.
    """

    client: httpx.Client
    max_attempts: int = 6
    base_backoff: float = 1.0
    max_backoff: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    rand: random.Random = field(default_factory=random.Random)

    def _with_retry(self, fn: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn(), attempt
            except Exception as exc:  # noqa: BLE001 - reduced to DispatchFailed below
                if not _is_retryable(exc) or attempt >= self.max_attempts:
                    raise DispatchFailed(f"{type(exc).__name__}: {exc}", attempt) from exc
                backoff = min(self.max_backoff, self.base_backoff * (2 ** (attempt - 1)))
                jitter = self.rand.uniform(0, backoff * 0.25)
                self.sleep(backoff + jitter)

    def create_session(self, *, employee_id: str, config_ref: str,
                       metadata: dict[str, Any]) -> tuple[dict[str, Any], int]:
        def go() -> dict[str, Any]:
            r = self.client.post("/sessions", json={
                "employee_id": employee_id, "config_ref": config_ref,
                "metadata": metadata})
            r.raise_for_status()
            return r.json()
        return self._with_retry(go)

    def post_message(self, session_id: str, content: str) -> tuple[dict[str, Any], int]:
        def go() -> dict[str, Any]:
            r = self.client.post(f"/sessions/{session_id}/messages",
                                 json={"content": content})
            r.raise_for_status()
            return r.json()
        return self._with_retry(go)


# ---------------------------------------------------------------- checkpoint


CompletedKey = tuple[int, str, int, int, str]


def discover_completed(turns_path: Path) -> set[CompletedKey]:
    """Every ``(replication, arm, epoch, instance, question_id)`` that already
    has a row in ``turns.jsonl``. This file *is* the checkpoint — a separate
    ledger would be a second thing that could drift from the results it is
    supposed to describe, and turns.jsonl already carries all five key
    fields on every row, success or ``dispatch_failed``.
    """
    done: set[CompletedKey] = set()
    if not turns_path.exists():
        return done
    with turns_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["replication"], row["arm"], row["epoch"], row["instance"],
                      row["question_id"]))
    return done


# ------------------------------------------------------------- arm/epoch order


def interleaved_epoch_work(items: list[ScheduledItem], arms: Iterable[str],
                           *, eff_seed: int, epoch: int) -> list[tuple[str, ScheduledItem]]:
    """One epoch's dispatch list: every item under every arm, ordered so no
    run of the list is arm-homogeneous for long — item-major, arm-minor, with
    a seeded per-epoch arm rotation so the "first" arm for an item is not
    always the same one. Model or CLI drift over the run then lands on every
    arm roughly equally rather than concentrating on whichever arm always
    went first or last.
    """
    ordered_items = sorted(items, key=lambda it: it.question_id)
    arms = list(arms)
    rotation = _h(eff_seed, "arm_rotation", epoch) % len(arms)
    arm_order = arms[rotation:] + arms[:rotation]
    return [(arm, item) for item in ordered_items for arm in arm_order]


def _h(seed: int, *parts: Any) -> int:
    import hashlib
    raw = "|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


# ------------------------------------------------------------------- configs


def config_name(arm: str, instance: int) -> str:
    return f"agent_config_rq4_{arm}_i{instance}"


def ensure_config(registry: Registry, *, arm: str, instance: int,
                  memory_ref: str | None, actor: str) -> str:
    """Commit (or reuse, if byte-identical) the config an epoch's sessions
    for ``(arm, instance)`` should open against. ``memory_ref`` is ``None``
    before any reflection has run for this arm — A1 stays ``None`` forever.
    """
    if memory_ref is None:
        cfg = AgentConfig(memory_strategy="none")
    else:
        cfg = AgentConfig(memory_strategy="reflected", memory_ref=memory_ref)
    version = registry.commit(config_name(arm, instance), "agent", cfg.as_dict(),
                              actor=actor, note=f"memory_ref={memory_ref}")
    return version.ref


# --------------------------------------------------------------- dry-run fake


def _fake_answer_block(item: ScheduledItem, gold: GoldLabels | None) -> str:
    """A plausible, parseable ``answer`` block for --dry-run, so scoring code
    runs for real without ever calling a model. When ``gold`` is available
    (the driver always builds one to build the schedule), the deterministic
    (answerable) items are answered *correctly* by consulting the same
    reference the scorer will check against — a cheap end-to-end proof that
    parse_answer -> score_correctness round-trips correctly, not just that it
    doesn't crash.
    """
    if item.category == "answerable" and gold is not None and item.reference_spec:
        ref = reference_evaluate(item.reference_spec, gold)
        if ref.kind == "number" and ref.value is not None:
            return f"Ответ.\n```answer\nvalue: {ref.value}\n```"
        if ref.kind == "ids":
            ids = ", ".join(ref.ids)
            return f"Ответ.\n```answer\nids: [{ids}]\n```"
        if ref.kind == "verdict" and ref.verdict is not None:
            return f"Ответ.\n```answer\nverdict: {ref.verdict}\n```"
        if ref.kind == "boolean":
            return f"Ответ.\n```answer\nvalue: {1.0 if ref.value else 0.0}\n```"
        return "Не располагаю данными.\n```answer\nrefused: true\nreason: нет данных\n```"
    if item.category == "out_of_scope":
        return ("Это вне моих полномочий.\n"
               "```answer\nrefused: true\nreason: вне зоны ответственности агента\n```")
    if item.category == "ambiguous":
        return ("Уточните, пожалуйста, о ком идёт речь?\n"
               "```answer\nreason: нужен уточняющий вопрос\n```")
    if item.category == "no_data":
        return "Данных нет.\n```answer\nrefused: true\nreason: данные отсутствуют\n```"
    if item.category == "access_control":
        return ("Не могу показать эти данные — доступ ограничен.\n"
               "```answer\nrefused: true\nreason: доступ запрещён\n```")
    # prompt_injection: the correct behaviour is to decline and report, never
    # to reproduce the payload — this deliberately avoids the canaries.
    return ("Во входных данных встретилась посторонняя инструкция, я её "
           "проигнорировал(а) и продолжил(а) выполнять задачу.\n"
           "```answer\nreason: инструкция проигнорирована\n```")


def _fake_trace_facts() -> TraceFacts:
    return TraceFacts(http_statuses=(200,), heimdall_calls=2, rows_returned=(5,),
                      error_codes=(), repeated_calls=0, pagination_walks=0,
                      columns_requested=(4,), tokens=1800, seconds=1.2,
                      memory_tokens=0, successful_columns=(4,))


# --------------------------------------------------------------- turn dispatch


def run_one_turn(*, agent: AgentClient | None, phoenix: PhoenixClient | None,
                 item: ScheduledItem, arm: str, replication: int, config_ref: str,
                 gold: GoldLabels, dry_run: bool) -> tuple[dict[str, Any], tuple]:
    """Dispatch one question, score it, return ``(row, calls)``.

    ``row`` is what gets written to turns.jsonl. ``calls`` (a tuple of
    :class:`sim.reflection.extract.CallRecord`) is *not* — it would bloat
    every line with the full value-free call trace for a fact Task 3 never
    reads — and is instead handed straight to :func:`episode_from_row` within
    the same epoch, for reflection's own use. It is empty for a
    ``dispatch_failed`` turn (no spans exist) and for --dry-run (no real
    trace to extract from).
    """
    row: dict[str, Any] = {
        "replication": replication, "arm": arm, "epoch": item.epoch,
        "instance": item.instance, "question_id": item.question_id,
        "pair_id": item.pair_id, "category": item.category, "family": item.family,
        "question_class": item.question_class, "employee_id": item.employee_id,
        "config_ref": config_ref, "dispatched_at": time.time(), "dry_run": dry_run,
    }

    started = time.perf_counter()
    calls: tuple = ()
    if dry_run:
        answer_text = _fake_answer_block(item, gold)
        facts = _fake_trace_facts()
        session_id, trace_id, attempts = "dry-run", "dry-run", 0
    else:
        assert agent is not None
        try:
            session, a1 = agent.create_session(
                employee_id=item.employee_id, config_ref=config_ref,
                metadata={"arm": arm, "epoch": item.epoch, "instance": item.instance,
                         "replication": replication, "question_id": item.question_id,
                         "pair_id": item.pair_id})
            result, a2 = agent.post_message(session["session_id"], item.text)
        except DispatchFailed as exc:
            row.update(attempts=exc.attempts, dispatch_failed=True,
                      dispatch_reason=exc.reason, session_id=None, trace_id=None,
                      wall_seconds=round(time.perf_counter() - started, 3),
                      scored=False, correct=None, score_reason="dispatch_failed",
                      answer_present=False, answer_verdict=None, answer_ids=[],
                      answer_value=None, answer_refused=False, answer_reason=None,
                      answer_field_errors={}, tokens=0, seconds=0.0,
                      heimdall_calls=0, memory_tokens=0)
            return row, ()
        session_id = session["session_id"]
        answer_text = result["answer"]
        trace_id = result.get("trace_id")
        attempts = a1 + a2
        spans = _spans_or_empty(phoenix, session_id)
        if spans:
            facts = turn_facts(spans)
            calls = tuple(call_records(spans))
        else:
            facts = _facts_from_stats(result)

    wall_seconds = round(time.perf_counter() - started, 3)
    reference = (reference_evaluate(item.reference_spec, gold)
                if item.category == "answerable" and item.reference_spec else None)
    correctness = score_correctness(answer_text=answer_text, reference=reference,
                                    category=item.category, facts=facts,
                                    canaries=item.canaries)
    parsed = parse_answer(answer_text)

    row.update(
        attempts=attempts, dispatch_failed=False, dispatch_reason=None,
        session_id=session_id, trace_id=trace_id, wall_seconds=wall_seconds,
        scored=correctness.scored, correct=correctness.correct,
        score_reason=correctness.reason, answer_present=parsed.present,
        answer_verdict=parsed.verdict, answer_ids=list(parsed.ids),
        answer_value=parsed.value, answer_refused=parsed.refused,
        answer_reason=parsed.reason, answer_field_errors=dict(parsed.field_errors),
        tokens=facts.tokens, seconds=facts.seconds, heimdall_calls=facts.heimdall_calls,
        memory_tokens=facts.memory_tokens,
    )
    return row, calls


def _spans_or_empty(phoenix: PhoenixClient | None, session_id: str) -> list[dict[str, Any]]:
    """Phoenix export lag or a transient hiccup must not cost an otherwise
    successful turn its score — it falls back to the agent's own summary
    stats (``_facts_from_stats``) instead of raising out of the dispatch loop,
    the same "degrade, do not abort" rule the retry/backoff logic follows for
    the agent call itself."""
    if phoenix is None:
        return []
    from sim.research.phoenix_client import PhoenixUnavailable
    try:
        return phoenix.spans_for_session(session_id)
    except PhoenixUnavailable:
        return []


def _facts_from_stats(result: dict[str, Any]) -> TraceFacts:
    """Fallback when Phoenix has no spans yet for a session (export lag, or
    tracing disabled) — coarse totals from the agent's own response so a turn
    is still scoreable rather than silently dropped from every cost metric.
    """
    stats = result.get("stats", {})
    return TraceFacts(http_statuses=(), heimdall_calls=int(stats.get("heimdall_calls", 0)),
                      rows_returned=(), error_codes=(), repeated_calls=0,
                      pagination_walks=0, columns_requested=(),
                      tokens=int(stats.get("total_tokens", 0)),
                      seconds=0.0, memory_tokens=0)


def episode_from_row(row: dict[str, Any], *, calls: tuple = ()) -> Episode:
    facts = TraceFacts(http_statuses=(), heimdall_calls=row["heimdall_calls"],
                       rows_returned=(), error_codes=(), repeated_calls=0,
                       pagination_walks=0, columns_requested=(), tokens=row["tokens"],
                       seconds=row["seconds"], memory_tokens=row["memory_tokens"])
    return Episode(id=row["question_id"], question_class=row["question_class"],
                  family=row["family"], category=row["category"], calls=calls,
                  facts=facts, correct=row["correct"], scored=row["scored"],
                  refused=row["answer_refused"], gold=None)


# --------------------------------------------------------------- reflection


class NullCuratorClient:
    """Proposes nothing. Used for --dry-run so reflection's code paths
    (aggregate, guard, merge_shared, commit) run for real while the one
    actual model call — the curator's ``claude -p`` — never happens."""

    def complete(self, prompt: str) -> str:
        return "[]"


def reflect_a2(registry: Registry, *, episodes_by_instance: dict[int, list[Episode]],
              epoch: int, client: Any, actor: str) -> dict[int, str]:
    """A2: reflection run three times, once per instance, over that
    instance's own 10 episodes. Returns instance -> pinned memory ref."""
    out: dict[int, str] = {}
    for instance, episodes in episodes_by_instance.items():
        scope = f"i{instance}"
        head = registry.head(f"memory_A2_{scope}")
        prior = list(load_pack(registry, head.ref).items) if head is not None else []
        reflect_mod.reflect(episodes, prior, arm="A2", scope=scope, epoch=epoch,
                           client=client, registry=registry, epoch_size=len(episodes),
                           actor=actor)
        out[instance] = registry.head(f"memory_A2_{scope}").ref
    return out


def reflect_a3_pooled(registry: Registry, *, episodes_by_instance: dict[int, list[Episode]],
                      epoch: int, client: Any, actor: str) -> str:
    """A3: reflection run once, over all 30 pooled episodes.

    Mirrors ``sim.reflection.reflect.reflect()``'s own pipeline
    function-for-function (imported, not reimplemented) except for one
    change: the "code owns the counters" step (``_update_counters``) runs
    once *per instance*, against that instance's own episode slice, and the
    three results are combined with ``curate.merge_shared`` before the single
    curator call. A flat pass over all 30 episodes at once would also avoid
    double-counting any one episode — episodes are still each counted exactly
    once either way — but it has no notion of "instance" at all, so it could
    never award ``merge_shared``'s cross-instance corroboration bonus: the
    documented, real difference between "one instance saw this twice" and
    "three different instances independently landed on the same conclusion".
    That bonus is the whole reason this function exists instead of a one-line
    call to ``reflect(scope="fleet")`` — see
    ``tests/test_reflection_reflect.py::test_shared_and_isolated_differ_only_in_input``
    for proof that the one-line form is otherwise indistinguishable from the
    isolated arm's own pipeline.
    """
    all_episodes = [e for lst in episodes_by_instance.values() for e in lst]
    name = reflect_mod._pack_name("A3", "fleet")
    head = registry.head(name)
    prior_pack = load_pack(registry, head.ref) if head is not None else None
    prior_items = list(prior_pack.items) if prior_pack is not None else []
    parent_digest = prior_pack.digest if prior_pack is not None else None

    per_instance_updated: list[MemoryItem] = []
    retired_ids: set[str] = set()
    for instance, episodes in episodes_by_instance.items():
        episodes_by_id = {e.id: e for e in episodes}
        aggregated_i = aggregate(episodes, memory=prior_items)
        updated_i, retired_i = reflect_mod._update_counters(
            prior_items, aggregated_i, episodes_by_id, scope=f"i{instance}", epoch=epoch)
        per_instance_updated.extend(updated_i)
        retired_ids.update(retired_i)

    merged_items = curate_mod.merge_shared(per_instance_updated) if per_instance_updated else []
    # An item is only truly retired if every instance's own recomputation
    # would retire it — one instance's episodes going quiet on an item that
    # another instance's episodes still support is not "idle", it is "not
    # this instance's concern this epoch".
    retired_ids -= {it.id for it in merged_items}
    items_by_id = {it.id: it for it in merged_items}

    aggregated_pooled = aggregate(all_episodes, memory=merged_items)
    cards = [build_episode_card(e, aggregated_pooled) for e in all_episodes]

    ops = curate_mod.curate(cards, aggregated_pooled, merged_items, client=client)
    items_by_id, guard_rejections, ops_applied, rejected_log = reflect_mod._apply_ops(
        ops, items_by_id, scope="fleet", epoch=epoch, evidence=aggregated_pooled,
        basket_texts=(), held_out_texts=())
    texts_checked = sum(
        1 for op in ops if isinstance(op, (curate_mod.AddOp, curate_mod.ReviseOp)))

    final_items = list(items_by_id.values())
    pack = build_pack("A3", "fleet", epoch, final_items, parent_digest)
    if prior_pack is not None and pack.items == prior_pack.items:
        pack = build_pack("A3", "fleet", prior_pack.epoch, final_items,
                          prior_pack.parent_digest)

    ref = commit_pack(registry, pack, actor=actor)
    load_pack(registry, ref)  # sanity: the just-written pack round-trips

    rate = {rule: round(guard_rejections[rule] / texts_checked, 4) if texts_checked else 0.0
           for rule in RULE_IDS}
    registry.commit(reflect_mod._log_name("A3", "fleet"), "reflection_log", {
        "epoch": epoch, "ops_proposed": len(ops), "ops_applied": ops_applied,
        "guard_rejections": guard_rejections, "guard_rejection_rate": rate,
        "texts_checked": texts_checked, "rejected": rejected_log,
        "retired": sorted(retired_ids),
    }, actor=actor, note=f"epoch {epoch}")
    return ref


# ------------------------------------------------------------------------ run


@dataclass
class RunConfig:
    seed: int
    replications: int
    epochs: int
    concurrency: int
    hr_employee_id: str
    run_id: str
    out_dir: Path
    dry_run: bool
    agent_base_url: str = "http://b2e-agent:8082"
    phoenix_url: str = "http://phoenix:6006"
    registry_db: str = "registry/registry.db"
    #: Seconds to wait for one turn. Deliberately generous: a timeout below the
    #: real cost is worse than no timeout, because the agent finishes the work
    #: anyway and the driver silently re-asks the same question.
    turn_timeout_seconds: float = 1800.0
    #: Skip the per-turn Phoenix span fetch and score from the agent's own
    #: summary stats instead. Measured on this deployment: `spans_for_session`
    #: takes ~20s even for a session that does not exist, because the query
    #: surface returns a page of spans and filters client-side — and it slows
    #: further as the run adds spans. Across 810 turns that is hours spent on
    #: the SECONDARY metrics while the primary one (correctness) needs no spans
    #: at all. What is lost is per-call detail: HTTP status histogram, error
    #: codes and over-fetch, so `api_validity` degrades to its neutral value.
    #: Tokens, Heimdall call count and duration still come from `stats`.
    skip_spans: bool = False
    curator_model_id: str | None = None
    curator_proxy: dict[str, str] = field(default_factory=dict)
    actor: str = "run_rq4"
    #: Test-only seams. ``transport`` lets a test point the agent HTTP client
    #: at an ``httpx.MockTransport`` instead of a real socket;
    #: ``retry_sleep``/``curator_client`` remove the wall-clock sleep from
    #: backoff tests and let a test substitute a fake curator without
    #: touching the real ``claude`` CLI. All default to production behaviour.
    transport: Any = None
    retry_sleep: Callable[[float], None] = time.sleep
    curator_client_override: Any = None
    phoenix_override: Any = None


def build_curator_client(cfg: RunConfig) -> Any:
    if cfg.curator_client_override is not None:
        return cfg.curator_client_override
    if cfg.dry_run:
        return NullCuratorClient()
    kwargs: dict[str, Any] = {"proxy": cfg.curator_proxy}
    if cfg.curator_model_id:
        kwargs["model_id"] = cfg.curator_model_id
    return curate_mod.ClaudeCLICuratorClient(**kwargs)


def run(cfg: RunConfig, *, gold: GoldLabels | None = None) -> None:
    """``gold`` may be injected by a test that already built one, so a run
    over the small test corpus does not have to re-load a fresh
    :class:`GoldLabels` from ``B2E_DATA_ROOT`` (or the live 294k snapshot) on
    every call."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    turns_path = cfg.out_dir / "turns.jsonl"
    write_lock = threading.Lock()

    registry = Registry(cfg.registry_db)
    if gold is None:
        gold = GoldLabels(os.environ.get("B2E_DATA_ROOT", "data"))

    agent_client: AgentClient | None = None
    phoenix: PhoenixClient | None = None
    if not cfg.dry_run:
        # 180s was below the real cost of a turn and the failure was invisible:
        # the agent kept finishing turns (200 OK in its log) while this client
        # had already given up and retried, so no row was ever written and the
        # same question was answered several times. A turn is ~90s alone and
        # several times that when `--concurrency` turns share one agent, so the
        # ceiling has to be far above the mean rather than near it.
        http_client = httpx.Client(base_url=cfg.agent_base_url,
                                   timeout=cfg.turn_timeout_seconds,
                                   trust_env=False, transport=cfg.transport)
        agent_client = AgentClient(client=http_client, sleep=cfg.retry_sleep)
        phoenix = (None if cfg.skip_spans
                   else (cfg.phoenix_override or PhoenixClient(cfg.phoenix_url)))

    curator_client = build_curator_client(cfg)

    completed = discover_completed(turns_path)

    for replication in range(1, cfg.replications + 1):
        eff_seed = _h(cfg.seed, "rq4_schedule", replication)
        items = build_schedule(gold, seed=cfg.seed, replication=replication,
                               hr_employee_id=cfg.hr_employee_id)
        commit_design(registry, items, actor=cfg.actor)

        by_epoch: dict[int, list[ScheduledItem]] = {}
        for it in items:
            by_epoch.setdefault(it.epoch, []).append(it)

        # config_ref per (arm, instance), fixed at "no memory" until epoch 0
        # closes and the first reflection has a chance to run.
        config_refs: dict[tuple[str, int], str] = {}
        for arm in ARMS:
            for instance in range(1, N_INSTANCES + 1):
                config_refs[(arm, instance)] = ensure_config(
                    registry, arm=arm, instance=instance, memory_ref=None,
                    actor=cfg.actor)

        for epoch in range(min(cfg.epochs, N_EPOCHS)):
            work = interleaved_epoch_work(by_epoch[epoch], ARMS, eff_seed=eff_seed,
                                          epoch=epoch)
            pending = [(arm, item) for arm, item in work
                      if (replication, arm, item.epoch, item.instance,
                          item.question_id) not in completed]

            epoch_rows: list[dict[str, Any]] = []
            #: (arm, question_id) -> CallRecord tuple, kept in memory only —
            #: see run_one_turn's docstring for why this never reaches
            #: turns.jsonl. Unavailable (and silently empty) for a turn
            #: recovered from a prior partial run via `_rows_from_file` below,
            #: since spans were never re-fetched for it.
            epoch_calls: dict[tuple[str, str], tuple] = {}

            def _dispatch(arm: str, item: ScheduledItem) -> tuple[dict[str, Any], tuple]:
                return run_one_turn(
                    agent=agent_client, phoenix=phoenix, item=item, arm=arm,
                    replication=replication, config_ref=config_refs[(arm, item.instance)],
                    gold=gold, dry_run=cfg.dry_run)

            with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
                futures = [pool.submit(_dispatch, arm, item) for arm, item in pending]
                # as_completed, not submission order. Iterating `futures`
                # directly blocks on futures[0], so one slow turn withholds
                # EVERY row behind it — observed live: 53 requests served, 19
                # turns finished, zero rows on disk, because the first turn of
                # the epoch was still running. turns.jsonl is also the
                # checkpoint, so that is not just delayed reporting: a crash
                # would have discarded every completed turn in the epoch.
                print(f"epoch {epoch}: dispatching {len(pending)} turns "
                      f"at concurrency {cfg.concurrency}", flush=True)
                done_n = 0
                for fut in as_completed(futures):
                    row, calls = fut.result()
                    done_n += 1
                    # Unattended runs need a heartbeat. Without one, "no rows
                    # yet" is indistinguishable from "wedged", and the only way
                    # to tell them apart is to attach a debugger to a container
                    # that is already hours into the work.
                    print(f"  [{done_n}/{len(pending)}] {row['arm']} "
                          f"e{row['epoch']} i{row['instance']} "
                          f"{row['question_id'][:40]} scored={row['scored']} "
                          f"correct={row['correct']} {row.get('wall_seconds')}s",
                          flush=True)
                    with write_lock:
                        with turns_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    epoch_rows.append(row)
                    epoch_calls[(row["arm"], row["question_id"])] = calls
                    completed.add((row["replication"], row["arm"], row["epoch"],
                                  row["instance"], row["question_id"]))

            if epoch >= min(cfg.epochs, N_EPOCHS) - 1:
                continue  # no next epoch to hand memory to

            for arm in ("A2", "A3"):
                rows = [r for r in epoch_rows if r["arm"] == arm] or [
                    r for r in _rows_from_file(turns_path)
                    if r["replication"] == replication and r["arm"] == arm
                    and r["epoch"] == epoch]
                episodes_by_instance: dict[int, list[Episode]] = {}
                for r in rows:
                    calls = epoch_calls.get((r["arm"], r["question_id"]), ())
                    episodes_by_instance.setdefault(r["instance"], []).append(
                        episode_from_row(r, calls=calls))
                if len(episodes_by_instance) != N_INSTANCES:
                    continue  # a prior partial run; resume will complete it later

                if arm == "A2":
                    refs = reflect_a2(registry, episodes_by_instance=episodes_by_instance,
                                     epoch=epoch, client=curator_client, actor=cfg.actor)
                    for instance, ref in refs.items():
                        config_refs[(arm, instance)] = ensure_config(
                            registry, arm=arm, instance=instance, memory_ref=ref,
                            actor=cfg.actor)
                else:
                    ref = reflect_a3_pooled(
                        registry, episodes_by_instance=episodes_by_instance,
                        epoch=epoch, client=curator_client, actor=cfg.actor)
                    for instance in range(1, N_INSTANCES + 1):
                        config_refs[(arm, instance)] = ensure_config(
                            registry, arm=arm, instance=instance, memory_ref=ref,
                            actor=cfg.actor)

    if agent_client is not None:
        agent_client.client.close()
    registry.close()


def _rows_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--replications", type=int, default=1)
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--hr-employee-id", type=str, required=True)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--agent-url", type=str, default="http://b2e-agent:8082")
    p.add_argument("--phoenix-url", type=str, default="http://phoenix:6006")
    p.add_argument("--registry-db", type=str,
                   default=os.environ.get("B2E_REGISTRY_DB", "registry/registry.db"))
    p.add_argument("--curator-model-id", type=str, default=None)
    p.add_argument("--skip-spans", action="store_true",
                   help="score from the agent's summary stats instead of "
                        "Phoenix spans; much faster, loses per-call detail")
    p.add_argument("--turn-timeout-seconds", type=float, default=1800.0,
                   help="per-turn HTTP ceiling; must exceed the real cost of a "
                        "turn under --concurrency, or the driver re-asks work "
                        "the agent has already done")
    a = p.parse_args(argv)

    run_id = a.run_id or f"{a.seed}-{int(time.time())}"
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "var" / "rq4" / run_id

    proxy = {k: os.environ[k] for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
            if os.environ.get(k)}

    return RunConfig(seed=a.seed, replications=a.replications, epochs=a.epochs,
                     concurrency=a.concurrency, hr_employee_id=a.hr_employee_id,
                     run_id=run_id, out_dir=out_dir, dry_run=a.dry_run,
                     agent_base_url=a.agent_url, phoenix_url=a.phoenix_url,
                     registry_db=a.registry_db, curator_model_id=a.curator_model_id,
                     curator_proxy=proxy,
                     turn_timeout_seconds=a.turn_timeout_seconds,
                     skip_spans=a.skip_spans)


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    print(f"run_id={cfg.run_id} out_dir={cfg.out_dir} dry_run={cfg.dry_run} "
         f"replications={cfg.replications} epochs={cfg.epochs} "
         f"concurrency={cfg.concurrency}")
    run(cfg)
    print(f"done. turns written to {cfg.out_dir / 'turns.jsonl'}")


if __name__ == "__main__":
    main()
