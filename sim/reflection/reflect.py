"""The reflection procedure. One pipeline; the isolated and shared arms
differ only in the episodes and prior memory handed to it.

Why that is the property this module is built around
--------------------------------------------------------
The whole point of running A2 (isolated) against A3 (shared) is to attribute
whatever difference shows up to *pooling across instances*. If ``reflect()``
itself branched on ``scope`` — a different prompt, a different cap, a
different ranking rule for ``scope == "fleet"`` — the shared arm would carry a
second, undeclared treatment, and the measured gap would be the sum of
"shared memory" and "whatever the branch did" with no way to tell the two
apart. So every step below (aggregate, build cards, curate, guard, count,
budget, render, commit) reads only ``episodes``, ``prior_items`` and
``epoch``; ``scope`` is written into the output labels and nowhere else. See
``tests/test_reflection_reflect.py::test_shared_and_isolated_differ_only_in_input``,
which calls this function twice with identical episodes/prior items/client and
asserts the two resulting packs render byte-identical text.

Two guards this module owns that Tasks 1-4 could not
---------------------------------------------------------
Neither exists anywhere upstream because neither is decidable without the
*procedure* itself:

* **Never run mid-epoch.** Activating a memory version while an epoch's
  episodes are still arriving would change ``agent_config_version`` (via
  ``AgentConfig.memory_ref``) underneath turns that are still in flight for
  that very epoch, corrupting the comparison the run design promises. The
  actual barrier — waiting for three live instances to close an epoch — is a
  driver concern the plan's self-review defers to Plan C. What belongs here
  is the refusal: ``reflect`` takes a required ``epoch_size`` (no default, by
  the same "a declared-but-inert condition is worse than a crash" rule
  ``sim.agent.config`` follows) and raises if the episode count handed to it
  does not match. A driver whose barrier has a bug still cannot make this
  function commit a partial epoch.
* **A night that learns nothing must not bump the registry version.**
  ``sim.reflection.memory.MemoryPack.epoch`` is part of the object
  ``commit_pack`` writes, so two packs from consecutive epochs that are
  otherwise identical still differ in that one field, and
  ``Registry.commit``'s byte-identical check would treat them as different
  content and mint a new version anyway — turning every reflection into a
  version bump regardless of whether anything was actually learned, which is
  exactly the "manufactures a condition change out of a no-op" failure
  ``commit_pack``'s own docstring warns about. This module closes that gap
  itself: before committing, it compares the freshly selected item set
  against the current head's, and if they match, commits under the *head's*
  epoch number instead of the new one, so the blob really is byte-identical
  and ``Registry.commit``'s own idempotency is what fires. See the
  ``prior_pack`` comparison inside ``reflect()`` below.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from sim.agent.loop import estimate_tokens
from sim.reflection.aggregate import Episode, aggregate, build_episode_card
from sim.reflection.curate import AddOp, DropOp, Op, ReviseOp, curate
from sim.reflection.guard import RULE_IDS, check as guard_check
from sim.reflection.memory import IDLE_RETIRE, MemoryItem, MemoryPack, build_pack, \
    commit_pack, load_pack
from sim.registry import Registry, canonical_bytes, sha256_hex


def _tokens_of(text: str) -> int:
    """Same wrapper ``sim.reflection.memory`` uses, so an item minted here and
    one loaded back from the registry estimate tokens identically."""
    return estimate_tokens([{"content": text}])


def mint_id(kind: str, trigger: dict[str, Any], text: str) -> str:
    """A ``MemoryItem`` id that is a content hash of ``(kind, trigger, text)``
    — nothing else.

    This one choice is what makes both idempotent re-runs and
    ``curate.merge_shared`` work without coordination: two curator calls
    (this epoch and a re-derivation of it, or two different instances)
    proposing the literal same lesson produce the literal same id, so they
    collide and merge instead of duplicating. Excluding ``scope`` from the
    hash is deliberate for the same reason — an id that depended on scope
    could never collide across instances, and ``merge_shared``'s exact-key
    merge would never have anything to merge.
    """
    payload = {"kind": kind, "trigger": trigger, "text": text}
    return "m_" + sha256_hex(canonical_bytes(payload))[:16]


def _update_counters(prior_items: Sequence[MemoryItem], aggregated: dict[str, Any],
                     episodes_by_id: dict[str, Episode], *, scope: str,
                     epoch: int) -> tuple[list[MemoryItem], list[str]]:
    """Step 6's counter half: apply this epoch's episodes as evidence for
    every item that predates it.

    Deliberately keyed off ``aggregated["memory_audit"]`` rather than
    re-deriving trigger matches here — ``sim.reflection.aggregate`` is the one
    place that logic lives (Task 4), and re-implementing it here would risk
    the two drifting apart, which would make the guard-facing audit report
    and the actual counters it is supposed to describe disagree.

    Returns the carried-forward items (counters updated, ``scope`` re-stamped
    to the current call's) and the ids retired this epoch for idling
    ``IDLE_RETIRE`` epochs — a code-owned decision (see ``memory.IDLE_RETIRE``'s
    docstring), never something the curator's ops can override.
    """
    audit = aggregated["memory_audit"]
    updated: list[MemoryItem] = []
    retired: list[str] = []

    for item in prior_items:
        info = audit.get(item.id, {"episodes_matched": (), "episodes_followed": ()})
        matched = info["episodes_matched"]
        followed = info["episodes_followed"]

        newly_supporting = sorted(
            eid for eid in followed
            if episodes_by_id[eid].scored and episodes_by_id[eid].correct
        )
        newly_refuting = sorted(
            eid for eid in followed
            if episodes_by_id[eid].scored and not episodes_by_id[eid].correct
        )

        epochs_idle = 0 if matched else item.epochs_idle + 1
        if epochs_idle >= IDLE_RETIRE:
            retired.append(item.id)
            continue

        gained_evidence = bool(newly_supporting or newly_refuting)
        updated.append(replace(
            item,
            scope=scope,
            support=item.support + len(newly_supporting),
            refute=item.refute + len(newly_refuting),
            episodes_support=tuple(sorted(set(item.episodes_support) | set(newly_supporting))),
            episodes_refute=tuple(sorted(set(item.episodes_refute) | set(newly_refuting))),
            epochs_idle=epochs_idle,
            last_useful_epoch=epoch if gained_evidence else item.last_useful_epoch,
            origin_instances=tuple(sorted(set(item.origin_instances) | {scope})),
            tokens=_tokens_of(item.text),
        ))
    return updated, retired


def _apply_ops(ops: Sequence[Op], items_by_id: dict[str, MemoryItem], *,
               scope: str, epoch: int, evidence: dict[str, Any],
               basket_texts: Sequence[str], held_out_texts: Sequence[str],
               ) -> tuple[dict[str, MemoryItem], dict[str, int], int, list[dict[str, Any]]]:
    """Step 5 (guard) and the code-owned half of Step 4 (apply what survives).

    Mutates nothing in place — returns a new ``items_by_id`` — so a caller
    that wants to inspect the pre-op state (this function's own tests do)
    still can. ``guard.check`` runs here, not inside ``curate``, because its
    per-rule outcome is reflection's own bookkeeping (the rejection-rate log),
    not something the curation call itself should ever see — a curator that
    could observe which rule rejected its last text would start optimising
    against the rule.
    """
    items_by_id = dict(items_by_id)
    rejections = {rule: 0 for rule in RULE_IDS}
    applied = 0
    rejected_log: list[dict[str, Any]] = []

    def _guarded(op_name: str, text: str, extra: dict[str, Any]) -> bool:
        violated = guard_check(text, evidence=evidence, basket_texts=basket_texts,
                               held_out_texts=held_out_texts)
        if violated is None:
            return True
        rejections[violated] += 1
        rejected_log.append({"op": op_name, "rule": violated, **extra})
        return False

    for op in ops:
        if isinstance(op, AddOp):
            if not _guarded("ADD", op.text, {"kind": op.kind, "trigger": op.trigger}):
                continue
            new_id = mint_id(op.kind, op.trigger, op.text)
            if new_id in items_by_id:
                # The curator proposed a lesson that (by content hash) already
                # exists verbatim. Nothing to add; not an error.
                continue
            items_by_id[new_id] = MemoryItem(
                id=new_id, scope=scope, kind=op.kind, trigger=op.trigger,
                text=op.text, support=0, refute=0, episodes_support=(),
                episodes_refute=(), created_epoch=epoch, last_useful_epoch=epoch,
                epochs_idle=0, origin_instances=(scope,), tokens=_tokens_of(op.text),
            )
            applied += 1
        elif isinstance(op, ReviseOp):
            existing = items_by_id.get(op.id)
            if existing is None:
                continue
            if not _guarded("REVISE", op.text, {"id": op.id}):
                continue
            items_by_id[op.id] = replace(existing, text=op.text,
                                         tokens=_tokens_of(op.text))
            applied += 1
        elif isinstance(op, DropOp):
            if items_by_id.pop(op.id, None) is not None:
                applied += 1
        # An op that is none of the three known types cannot reach this loop:
        # curate.parse_ops only ever constructs AddOp/ReviseOp/DropOp.

    return items_by_id, rejections, applied, rejected_log


def _log_name(arm: str, scope: str) -> str:
    return f"reflection_log_{arm}_{scope}"


def _pack_name(arm: str, scope: str) -> str:
    return f"memory_{arm}_{scope}"


def reflect(episodes: Sequence[Episode], prior_items: Sequence[MemoryItem], *,
           arm: str, scope: str, epoch: int, client: Any, registry: Registry,
           epoch_size: int, actor: str = "reflection",
           basket_texts: Sequence[str] = (), held_out_texts: Sequence[str] = (),
           ) -> MemoryPack:
    """Run one epoch's reflection and commit the result.

    ``epoch_size`` is required, not defaulted, and is the whole of this
    function's "never mid-epoch" refusal — see the module docstring.
    ``basket_texts``/``held_out_texts`` feed guard rule G4 (question
    paraphrase); a caller with no basket to check against gets G4 as a
    permanent pass, which is honest about what was not checked rather than
    silently assumed safe.
    """
    if len(episodes) != epoch_size:
        raise ValueError(
            f"reflection refuses to run mid-epoch: epoch_size={epoch_size} but "
            f"{len(episodes)} episode(s) were handed to reflect() for arm={arm!r} "
            f"scope={scope!r} epoch={epoch}"
        )

    episodes_by_id = {e.id: e for e in episodes}
    aggregated = aggregate(episodes, memory=prior_items)
    cards = [build_episode_card(e, aggregated) for e in episodes]

    updated_items, retired = _update_counters(
        prior_items, aggregated, episodes_by_id, scope=scope, epoch=epoch)
    items_by_id = {it.id: it for it in updated_items}

    ops = curate(cards, aggregated, updated_items, client=client)
    items_by_id, guard_rejections, ops_applied, rejected_log = _apply_ops(
        ops, items_by_id, scope=scope, epoch=epoch, evidence=aggregated,
        basket_texts=basket_texts, held_out_texts=held_out_texts)

    # Every ADD/REVISE op is checked exactly once by `_guarded`, whether it
    # passes or is rejected, so the rejection-rate denominator is simply how
    # many such ops the curator proposed this epoch.
    texts_checked = sum(1 for op in ops if isinstance(op, (AddOp, ReviseOp)))

    final_items = list(items_by_id.values())

    name = _pack_name(arm, scope)
    head = registry.head(name)
    prior_pack = load_pack(registry, head.ref) if head is not None else None
    parent_digest = prior_pack.digest if prior_pack is not None else None

    pack = build_pack(arm, scope, epoch, final_items, parent_digest)
    if prior_pack is not None and pack.items == prior_pack.items:
        # See the module docstring: nothing this epoch changed the selected
        # item set, so commit under the prior epoch number too, making the
        # blob genuinely byte-identical and letting Registry.commit's own
        # idempotency be what decides not to bump the version.
        pack = build_pack(arm, scope, prior_pack.epoch, final_items,
                          prior_pack.parent_digest)

    ref = commit_pack(registry, pack, actor=actor)
    committed = load_pack(registry, ref)

    rate = {
        rule: round(guard_rejections[rule] / texts_checked, 4) if texts_checked else 0.0
        for rule in RULE_IDS
    }
    payload = {
        "epoch": epoch, "ops_proposed": len(ops), "ops_applied": ops_applied,
        "guard_rejections": guard_rejections, "guard_rejection_rate": rate,
        "texts_checked": texts_checked, "rejected": rejected_log, "retired": retired,
    }
    registry.commit(_log_name(arm, scope), "reflection_log", payload, actor=actor,
                    note=f"epoch {epoch}")

    return committed


def load_log(registry: Registry, arm: str, scope: str) -> dict[str, Any]:
    """The most recent epoch's guard-rejection report for one arm/scope.

    A thin convenience wrapper — ``registry.load`` already does the real
    work — so a caller (an admin view, a test) does not have to know the
    ``reflection_log_<arm>_<scope>`` naming convention this module owns.
    """
    _version, data = registry.load(_log_name(arm, scope))
    return data
