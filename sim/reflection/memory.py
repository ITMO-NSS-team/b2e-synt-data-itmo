"""The memory artefact: record shape, budget, deterministic renderer, storage.

What this module is not
------------------------
There is no LLM call anywhere below. ``sim/reflection/reflect.py`` (Task 5) is
where a model proposes lesson *text*; everything here — which items survive the
budget, what order they render in, what bytes land in the registry — is a pure
function of already-decided ``MemoryItem`` values. Keeping the split hard is
what makes the counters trustworthy: a renderer that could also decide "this
item worked" would let a formatting bug quietly become an evidence bug.

Why the registry and not a bespoke table
-----------------------------------------
``sim.registry.Registry`` already gives content addressing and an append-only
history for free, and both properties are load-bearing here specifically. A
memory version has to be nameable in a trace the same way a system prompt is
(``memory_<arm>_<scope>@N``), and a night that changes nothing must not look
like a night that changed something — which is exactly the guarantee
``Registry.commit`` makes for byte-identical content. Building a second store
that reimplemented both properties would be duplicating machinery this project
already trusts.

Why the budget check calls ``render`` instead of summing ``item.tokens``
-------------------------------------------------------------------------
Each item's own ``tokens`` field is bookkeeping written when the item was
minted (Task 5), and is *not* re-derived here — trusting a stored count would
be exactly the "second opinion" the spec forbids. ``select`` reserves capacity
for pitfalls using that stored estimate, because that decision only has to be
approximately right. But the hard cap — the thing a research write-up will cite
as "≤1200 tokens" — is checked by rendering the actual candidate set and
calling ``sim.agent.loop.estimate_tokens`` on the real string, the same
estimator every other token budget in this codebase uses. A per-item sum would
silently drift from that the moment a section header or the closing note ate
into the budget; rendering first and measuring what came out cannot drift by
construction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from sim.agent.loop import estimate_tokens
from sim.registry import Registry, sha256_hex

#: The three kinds a lesson can be. ``pitfall`` is load-bearing: it is the only
#: kind that can say "do not answer this", and a memory pack that can only ever
#: say "do more" is a refusal suppressor — the failure mode that would let a
#: memory arm win on answerable questions while quietly losing every caution
#: one. See ``PITFALL_RESERVE`` below.
MEMORY_KINDS = ("api_mechanic", "method", "pitfall")

#: Hard caps from the spec (§4). Every one of them is cited by the analysis
#: write-up, so they live at module scope rather than inside a function body —
#: a threshold buried in an expression cannot be quoted back to a reviewer.
K_MAX = 24
T_MAX = 1200
PITFALL_RESERVE = 300
#: An item that has not mattered for this many consecutive epochs is retired by
#: the reflection procedure (Task 5). Defined here, beside the other budget
#: constants, because it is part of the same "memory does not grow without
#: bound" contract even though nothing in this module enforces it directly.
IDLE_RETIRE = 3

#: Registry names the reflection worker must never produce. Guarded at the
#: level of the *identity fields* (``arm``, ``scope``) rather than only on the
#: assembled name: the name is always ``memory_<arm>_<scope>`` here, so an
#: assembled-name check alone could never fire, and the constraint this exists
#: to enforce — "the reflection worker must never commit a registry name
#: matching ^agent_config or ^system_prompt" — is about what the worker is
#: capable of, not about what today's one call site happens to produce.
_FORBIDDEN_NAME = re.compile(r"^(agent_config|system_prompt)")


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One lesson, plus the evidence code has collected for it.

    ``trigger`` is a structural pattern (question class, schema/model names —
    never a value) that the pre-aggregation step matches episodes against; it
    is what makes ``support``/``refute`` mean something more precise than "this
    memory pack was active during a passing turn". ``support`` and ``refute``
    are never written by the model that proposes ``text`` — see the module
    docstring — which is why ``curate.parse_ops`` (Task 5) rejects any op that
    carries them.
    """

    id: str
    scope: str
    kind: str
    trigger: dict[str, Any]
    text: str
    support: int
    refute: int
    episodes_support: tuple[str, ...]
    episodes_refute: tuple[str, ...]
    created_epoch: int
    last_useful_epoch: int
    epochs_idle: int
    origin_instances: tuple[str, ...]
    tokens: int

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"kind {self.kind!r} not in {MEMORY_KINDS}")


@dataclass(frozen=True, slots=True)
class MemoryPack:
    """The budgeted, rendered result for one arm/scope at one epoch.

    ``digest`` hashes the *rendered* text rather than the item list, because
    the rendered string is the only part of this object anything downstream —
    the prompt, the root-span attribute, a researcher diffing two epochs —
    actually touches. Two packs whose items differ only in a field that never
    reaches ``rendered`` (say, ``episodes_support`` growing by one id) are
    indistinguishable to the model and should be indistinguishable by digest
    too; hashing the item list would tell them apart for no observable reason.

    ``parent_digest`` is carried rather than looked up, so a pack is
    self-describing lineage without a second registry read.
    """

    arm: str
    scope: str
    epoch: int
    items: tuple[MemoryItem, ...]
    rendered: str
    digest: str
    parent_digest: str | None


def _tokens_of(text: str) -> int:
    """The one and only token estimate used anywhere in this module.

    Wraps ``sim.agent.loop.estimate_tokens`` in the shape it expects — a list
    of ``{"content": ...}`` messages — so every call site in this file spells
    the estimate the same way instead of each re-deriving the wrapper.
    """
    return estimate_tokens([{"content": text}])


def score(item: MemoryItem) -> float:
    """Laplace-smoothed pass rate: ``(support + 1) / (support + refute + 2)``.

    A raw ratio lets one lucky episode (1 support, 0 refute → 1.0) outrank nine
    of ten (9 support, 1 refute → 0.9), which is backwards — the second item
    has far more evidence behind the same conclusion. The +1/+2 pseudo-count
    pulls single-observation items toward 0.5 until more evidence arrives,
    which is what makes ``select`` below trust accumulated items over new ones.
    """
    return round((item.support + 1) / (item.support + item.refute + 2), 4)


def select(items: Sequence[MemoryItem], *, k_max: int = K_MAX, t_max: int = T_MAX,
          pitfall_reserve: int = PITFALL_RESERVE) -> list[MemoryItem]:
    """Tiered greedy pack: pitfalls get first claim on ``pitfall_reserve``
    tokens, then every remaining item competes by descending score.

    The tie-break on ``id`` in both tiers is not cosmetic: two items scoring
    identically would otherwise order however the input happened to be sorted,
    and a rerun of the same epoch with the same items in a different order
    would silently commit a different pack.
    """
    ranked = sorted(items, key=lambda it: (-score(it), it.id))
    pitfalls = [it for it in ranked if it.kind == "pitfall"]

    chosen: list[MemoryItem] = []
    chosen_ids: set[str] = set()

    def fits(candidate: MemoryItem) -> bool:
        if len(chosen) >= k_max:
            return False
        # The true check: render what the pack would be with this item added,
        # and measure that string. See the module docstring for why this is
        # not a sum of `item.tokens`.
        return _tokens_of(render(chosen + [candidate])) <= t_max

    pitfall_tokens = 0
    for item in pitfalls:
        if pitfall_tokens >= pitfall_reserve:
            break
        if not fits(item):
            continue
        chosen.append(item)
        chosen_ids.add(item.id)
        pitfall_tokens += item.tokens

    for item in ranked:
        if item.id in chosen_ids:
            continue
        if not fits(item):
            continue
        chosen.append(item)
        chosen_ids.add(item.id)

    return chosen


#: Section order and headers, fixed and Russian — this text reaches the agent.
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("pitfall", "### Чего не делать"),
    ("method", "### Приёмы"),
    ("api_mechanic", "### Заметки о витринах"),
)

#: Fixed closing line. States plainly that the block above is operational
#: observation, not an answer key — the same boundary the guard rules (Task 4)
#: enforce on the way in, restated on the way out so the model is told, not
#: just prevented.
_CLOSING_NOTE = (
    "Это наблюдения о том, как работать с API Heimdall, а не о том, каким "
    "должен быть ответ на вопрос."
)


def render(items: Sequence[MemoryItem]) -> str:
    """Deterministic f-string. Never an LLM — see the module docstring.

    Grouped by kind into the three fixed sections, each internally ordered by
    descending score with the same ``id`` tie-break ``select`` uses, so the
    output depends only on the *set* of items passed in, never on the order
    they arrived in. Only lesson text is emitted: no ``support``/``refute``,
    no ``id``, no episode reference. A rendered block that carried a counter
    would let the model read its own confidence and start asserting from it
    rather than from evidence code still owns.
    """
    by_kind: dict[str, list[MemoryItem]] = {kind: [] for kind, _ in _SECTIONS}
    for item in items:
        by_kind.setdefault(item.kind, []).append(item)

    blocks: list[str] = []
    for kind, header in _SECTIONS:
        bucket = sorted(by_kind.get(kind, ()), key=lambda it: (-score(it), it.id))
        if not bucket:
            continue
        lines = "\n".join(f"- {it.text}" for it in bucket)
        blocks.append(f"{header}\n{lines}")

    blocks.append(_CLOSING_NOTE)
    return "\n\n".join(blocks)


def build_pack(arm: str, scope: str, epoch: int, items: Sequence[MemoryItem],
              parent_digest: str | None) -> MemoryPack:
    """Apply the budget, render, hash. The only place a ``MemoryPack`` is made."""
    chosen = tuple(select(items))
    rendered = render(chosen)
    digest = sha256_hex(rendered.encode("utf-8"))
    return MemoryPack(arm=arm, scope=scope, epoch=epoch, items=chosen,
                      rendered=rendered, digest=digest, parent_digest=parent_digest)


# --------------------------------------------------------------- (de)serialise


def _item_to_dict(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id, "scope": item.scope, "kind": item.kind,
        "trigger": item.trigger, "text": item.text,
        "support": item.support, "refute": item.refute,
        "episodes_support": list(item.episodes_support),
        "episodes_refute": list(item.episodes_refute),
        "created_epoch": item.created_epoch,
        "last_useful_epoch": item.last_useful_epoch,
        "epochs_idle": item.epochs_idle,
        "origin_instances": list(item.origin_instances),
        "tokens": item.tokens,
    }


def _item_from_dict(data: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        id=str(data["id"]), scope=str(data["scope"]), kind=str(data["kind"]),
        trigger=dict(data.get("trigger") or {}), text=str(data["text"]),
        support=int(data["support"]), refute=int(data["refute"]),
        episodes_support=tuple(data.get("episodes_support") or ()),
        episodes_refute=tuple(data.get("episodes_refute") or ()),
        created_epoch=int(data["created_epoch"]),
        last_useful_epoch=int(data["last_useful_epoch"]),
        epochs_idle=int(data["epochs_idle"]),
        origin_instances=tuple(data.get("origin_instances") or ()),
        tokens=int(data["tokens"]),
    )


def _pack_to_dict(pack: MemoryPack) -> dict[str, Any]:
    return {
        "arm": pack.arm, "scope": pack.scope, "epoch": pack.epoch,
        "items": [_item_to_dict(it) for it in pack.items],
        "rendered": pack.rendered, "digest": pack.digest,
        "parent_digest": pack.parent_digest,
    }


def _pack_from_dict(data: dict[str, Any]) -> MemoryPack:
    return MemoryPack(
        arm=str(data["arm"]), scope=str(data["scope"]), epoch=int(data["epoch"]),
        items=tuple(_item_from_dict(d) for d in data.get("items") or ()),
        rendered=str(data["rendered"]), digest=str(data["digest"]),
        parent_digest=(None if data.get("parent_digest") is None
                       else str(data["parent_digest"])),
    )


def commit_pack(registry: Registry, pack: MemoryPack, *, actor: str) -> str:
    """Append ``pack`` to the registry, return its pinned ``name@N`` ref.

    Refuses rather than writing: ``arm`` and ``scope`` are checked, not the
    assembled name, because the name built two lines below is always
    ``memory_<arm>_<scope>`` and could therefore never itself start with
    ``agent_config`` or ``system_prompt`` — checking only the assembled string
    would make this assertion permanently unreachable and give a false sense of
    a guard that does nothing. Checking the identity fields closes the risk at
    its source: whatever future caller builds the name, a memory pack whose
    ``arm``/``scope`` collides with the reserved namespaces is refused before a
    single byte is written.

    Idempotent for free: ``Registry.commit`` returns the existing head
    unchanged when the content is byte-identical, so committing the same pack
    twice — or an epoch that changed nothing — never bumps the version.
    """
    for field_name, value in (("arm", pack.arm), ("scope", pack.scope)):
        assert not _FORBIDDEN_NAME.match(value), (
            f"reflection must never write into the agent_config/system_prompt "
            f"namespace; pack.{field_name}={value!r} would let a memory "
            f"version masquerade as one of those"
        )
    name = f"memory_{pack.arm}_{pack.scope}"
    version = registry.commit(
        name, "memory", _pack_to_dict(pack), actor=actor,
        note=f"epoch {pack.epoch}, {len(pack.items)} item(s)",
    )
    return version.ref


def load_pack(registry: Registry, ref: str) -> MemoryPack:
    """Resolve ``name`` or ``name@N`` and reconstruct the pack it names."""
    _version, data = registry.load(ref)
    return _pack_from_dict(data)
