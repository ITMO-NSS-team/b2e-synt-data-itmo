"""What the LLM curator (Task 5) is allowed to see: counters code owns, and
episode cards with no answer in them.

Why pre-aggregation exists at all
----------------------------------
"Which class fails most, which error keeps recurring, which memory item is
actually earning its keep" are all counting questions, and a model asked to
both notice a pattern and report how strong it is will round its own hunch up
— the failure Task 1's ``score()`` docstring calls "the model writing its own
evidence". Everything in this file is a pure aggregation over already-scored,
already-extracted facts (``sim.reflection.extract.CallRecord``,
``sim.research.evaluate.TraceFacts``/``CorrectnessResult``); nothing here
calls an LLM, and ``curate.py`` (Task 5) is the only place a proposed lesson's
*text* gets written — this module only ever produces counters and cards for
that call's input.

Why ``EpisodeCard`` never carries the gold value
---------------------------------------------------
Reflection has to learn *that* a turn failed and *what kind* of failure it
was, never *what the right answer would have been*. A card that carried
``reference.value`` would let the curator's LLM call quote it straight into a
lesson text, and a lesson that can carry a value is exactly the
answer-transport channel Task 3's ``query_shape`` closes on the query side —
``EpisodeCard`` closes the same channel on the scoring side. ``Episode`` (the
input to this module) is allowed to carry the gold value, because Plan A's
scorer produced it and something has to hold it long enough to compute
``correct``; the discipline is that nothing downstream of
``build_episode_card`` ever sees it again.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from sim.reflection.extract import CallRecord
from sim.reflection.memory import MemoryItem
from sim.research.evaluate import TraceFacts

#: The one error-key shape this whole module agrees on: which endpoint, what
#: it answered, what code it gave, and which argument names were present.
#: Never includes an argument *value* — ``CallRecord.argument_keys`` already
#: is names only, inherited from ``query_shape``.
ErrorKey = tuple[str, int | None, str, tuple[str, ...]]

_VERDICTS = ("PASS", "FAIL", "UNSCORED")


@dataclass(frozen=True, slots=True)
class Episode:
    """One scored (question, turn) pair, in the vocabulary reflection reads.

    ``calls`` and ``facts`` come from Task 3 (``extract.call_records`` /
    ``extract.turn_facts``); ``correct``/``scored``/``refused``/``gold`` come
    from Plan A's scorer (``sim.research.evaluate``). Assembling those two
    already-computed sources into one record is this type's whole job — it
    does no scoring and no extraction of its own.
    """

    id: str
    question_class: str
    family: str
    category: str
    calls: tuple[CallRecord, ...]
    facts: TraceFacts
    correct: bool | None
    scored: bool
    refused: bool
    #: The answer key for this question (a ``Reference``'s value/ids/verdict,
    #: or whatever shape a caller's scorer used). Present only so this module
    #: can decide pass/fail; never read again after ``build_episode_card``.
    gold: Any = None


@dataclass(frozen=True, slots=True)
class EpisodeCard:
    """A value-free, gold-free summary of one episode — the unit the Task 5
    curator's LLM call actually reads. See the module docstring for why
    ``gold`` (and any answer text) can never reach this type."""

    question_class: str
    family: str
    category: str
    #: Value-free, human-readable call trace, e.g.
    #: ``"mcp_query(dm_core.employee_actual, cols=3, rowwise)"``.
    plan: tuple[str, ...]
    #: ``[{"code": "unknown-column", "fixed_by": {"added": [...], "removed": [...]}}]``
    #: — ``fixed_by`` present only when a later call of the same shape in the
    #: same turn actually succeeded.
    api_errors: tuple[dict[str, Any], ...]
    verdict: str
    failure_class: str | None
    calls: int
    tokens: int
    seconds: float

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} not in {_VERDICTS}")


# ------------------------------------------------------------------- plan text


def _plan_step(call: CallRecord) -> str:
    """One call, rendered the way the spec's own example does — target, then
    a couple of value-free shape hints. Never a value: only counts
    (``cols=3``), buckets (``limit_bucket``) and key *names*
    (``argument_keys``), all of which are already what ``query_shape`` kept."""
    if call.tool != "mcp_query":
        return call.tool
    target = f"{call.schema}.{call.logic_model}" if (call.schema or call.logic_model) \
        else "?"
    bits = [target]
    if call.columns:
        bits.append(f"cols={len(call.columns)}")
        bits.append("rowwise")
    else:
        bits.append("aggregate")
    if "offset" in call.argument_keys:
        bits.append("offset")
    return f"mcp_query({', '.join(bits)})"


# --------------------------------------------------------------- failure class

#: A coarse, code-decided taxonomy — deliberately not exhaustive. Reflection
#: needs *a* stable label to group episodes by more than it needs a perfect
#: one; Task 5's curator sees the label, never derives it, so refining this
#: taxonomy later never changes what the model is trusted to decide for
#: itself.
FAILURE_CLASSES = ("wrong_value", "unnecessary_refusal", "missed_refusal", "unscored")


def _failure_class(episode: Episode) -> str | None:
    if not episode.scored:
        return "unscored"
    if episode.correct:
        return None
    if episode.category == "answerable":
        return "unnecessary_refusal" if episode.refused else "wrong_value"
    # Every other category's correct behaviour is some form of not answering
    # outright (refuse, ask, decline) — see sim.research.evaluate's per-category
    # branches. A failure here is either "attempted anyway" or "declined for
    # the wrong reason"; without the parsed answer's full detail this module
    # can only distinguish the first from the second by which everyday
    # ``refused`` flag Plan A's parser set.
    return "missed_refusal" if not episode.refused else "unnecessary_refusal"


def _episode_errors(episode: Episode, fix_diffs: dict[ErrorKey, dict]) -> tuple[dict, ...]:
    out: list[dict[str, Any]] = []
    for call in episode.calls:
        if not call.error_code:
            continue
        entry: dict[str, Any] = {"code": call.error_code}
        diff = fix_diffs.get(_error_key(call))
        if diff is not None:
            entry["fixed_by"] = diff
        out.append(entry)
    return tuple(out)


def build_episode_card(episode: Episode, aggregated: dict[str, Any]) -> EpisodeCard:
    """The card for one episode, given ``aggregate()``'s output (for the
    error fix-diffs — see its docstring for why those are computed globally
    rather than per card)."""
    verdict = "UNSCORED" if not episode.scored else ("PASS" if episode.correct else "FAIL")
    return EpisodeCard(
        question_class=episode.question_class, family=episode.family,
        category=episode.category,
        plan=tuple(_plan_step(c) for c in episode.calls),
        api_errors=_episode_errors(episode, aggregated.get("_fix_diffs", {})),
        verdict=verdict,
        failure_class=_failure_class(episode) if verdict != "PASS" else None,
        calls=len(episode.calls), tokens=episode.facts.tokens,
        seconds=episode.facts.seconds,
    )


# ------------------------------------------------------------- error histogram


def _error_key(call: CallRecord) -> ErrorKey:
    return (call.tool, call.http_status, call.error_code or "", call.argument_keys)


def _fix_diffs_within(calls: Sequence[CallRecord]) -> dict[ErrorKey, dict[str, list[str]]]:
    """For each erroring call, what the next successful call *to the same
    target* in the same turn changed about its argument names.

    Restricted to one turn's own calls, not the whole episode set: "the fix"
    is the agent correcting itself mid-turn, which is a fact about that turn.
    Two different turns that both hit ``unknown-column`` and both eventually
    used a different column are two independent pieces of evidence for the
    same histogram bucket, not one bigger fix.
    """
    diffs: dict[ErrorKey, dict[str, list[str]]] = {}
    for i, call in enumerate(calls):
        if not call.error_code:
            continue
        key = _error_key(call)
        if key in diffs:
            continue
        target = (call.tool, call.schema, call.logic_model)
        for later in calls[i + 1:]:
            if (later.tool, later.schema, later.logic_model) != target:
                continue
            if later.http_status != 200:
                continue
            added = sorted(set(later.argument_keys) - set(call.argument_keys))
            removed = sorted(set(call.argument_keys) - set(later.argument_keys))
            diffs[key] = {"added": added, "removed": removed}
            break
    return diffs


def _error_histogram(episodes: Sequence[Episode],
                     fix_diffs: dict[ErrorKey, dict]) -> dict[str, dict[str, Any]]:
    counts: dict[ErrorKey, int] = defaultdict(int)
    for episode in episodes:
        for call in episode.calls:
            if call.error_code:
                counts[_error_key(call)] += 1
    # String keys: this dict is evidence handed to an LLM prompt (Task 5) and
    # to guard.check's G2 (which walks it for numbers) — both want something
    # that survives round-tripping through JSON, which a tuple key does not.
    return {
        f"{tool}|{status}|{code}|{','.join(keys)}": {
            "count": n, "fixed_by": fix_diffs.get((tool, status, code, keys)),
        }
        for (tool, status, code, keys), n in counts.items()
    }


# ---------------------------------------------------------------- per-class

def _per_class(episodes: Sequence[Episode]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        buckets[episode.question_class].append(episode)

    out: dict[str, dict[str, Any]] = {}
    for question_class, group in buckets.items():
        scored = [e for e in group if e.scored]
        out[question_class] = {
            "n": len(group),
            "pass_rate": (sum(1 for e in scored if e.correct) / len(scored)
                         if scored else None),
            "mean_calls": round(statistics.fmean(len(e.calls) for e in group), 2),
            "mean_tokens": round(statistics.fmean(e.facts.tokens for e in group), 1),
            "mean_seconds": round(statistics.fmean(e.facts.seconds for e in group), 2),
        }
    return out


# -------------------------------------------------------------------- waste

#: Same rationale as sim.research.evaluate's own floor: the leanest observed
#: query in a small episode set can be implausibly narrow, and a fixed floor
#: keeps a two-column probe from making every ten-column query look greedy.
_LEAN_COLUMNS_FLOOR = 8
_OVERFETCH_FACTOR = 3.0


def _waste(episodes: Sequence[Episode]) -> dict[str, int]:
    successful_widths = [
        len(c.columns) for e in episodes for c in e.calls
        if c.tool == "mcp_query" and c.http_status == 200 and c.rows > 0 and c.columns
    ]
    threshold = (max(min(successful_widths), _LEAN_COLUMNS_FLOOR) * _OVERFETCH_FACTOR
                if successful_widths else float("inf"))
    over_fetch = sum(
        1 for e in episodes for c in e.calls
        if c.tool == "mcp_query" and len(c.columns) > threshold
    )
    return {
        "repeated_calls": sum(e.facts.repeated_calls for e in episodes),
        "pagination_walks": sum(e.facts.pagination_walks for e in episodes),
        "over_fetch_calls": over_fetch,
    }


# --------------------------------------------------------------- memory audit


def _trigger_matches(trigger: dict[str, Any], episode: Episode) -> bool:
    """Whether ``episode`` is the kind of situation ``trigger`` describes.

    Unknown trigger keys are ignored rather than treated as a mismatch, so a
    trigger dimension added later (Task 5 may grow the vocabulary) does not
    retroactively zero out every existing item's audit the day it appears.
    """
    for key, allowed in (trigger or {}).items():
        allowed_set = set(allowed) if isinstance(allowed, (list, tuple, set)) else {allowed}
        if key == "question_class" and episode.question_class not in allowed_set:
            return False
        if key == "family" and episode.family not in allowed_set:
            return False
        if key == "schema_model":
            targets = {f"{c.schema}.{c.logic_model}" for c in episode.calls
                      if c.schema and c.logic_model}
            if not targets & allowed_set:
                return False
    return True


def _plan_followed(item: MemoryItem, episode: Episode) -> bool:
    """Whether the episode's own behaviour is consistent with this item.

    A ``pitfall`` item's trigger describes a situation that should end in
    refusal, so "followed" is observable directly from ``episode.refused``.
    ``method``/``api_mechanic`` items describe *how* to call something
    correctly, which this module cannot verify without re-deriving the
    "right" call shape — a trigger match is the best signal available without
    that, so those two kinds count as followed whenever they matched at all.
    This is a known simplification, not a claim that plan-following is fully
    verified; Task 5's curator sees ``episodes_matched`` and
    ``episodes_followed`` separately so a reviewer can tell the two apart.
    """
    if item.kind == "pitfall":
        return bool(episode.refused)
    return True


def _memory_audit(episodes: Sequence[Episode],
                  memory: Sequence[MemoryItem]) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for item in memory:
        matched = [e for e in episodes if _trigger_matches(item.trigger, e)]
        followed = [e for e in matched if _plan_followed(item, e)]
        scored_followed = [e for e in followed if e.scored]
        audit[item.id] = {
            "episodes_matched": tuple(e.id for e in matched),
            "episodes_followed": tuple(e.id for e in followed),
            "pass_rate_when_followed": (
                round(sum(1 for e in scored_followed if e.correct) / len(scored_followed), 4)
                if scored_followed else None),
        }
    return audit


# ------------------------------------------------------------------ aggregate


def aggregate(episodes: Sequence[Episode], *, memory: Sequence[MemoryItem] = ()) -> dict[str, Any]:
    """Step 2 of reflection: everything code can count before an LLM sees
    anything.

    ``memory`` (prior items to audit) is not in the plan's one-argument
    signature, but the memory audit it asks for — "for each existing item,
    which episodes its trigger matched" — has no existing item to audit
    without one; see the Task 3+4 report for this deviation and why an
    audit-less ``aggregate`` would silently drop a whole pre-aggregation
    output the plan itself lists. Defaults to ``()`` so a caller that has no
    prior memory (epoch 1) does not have to invent one.
    """
    fix_diffs: dict[ErrorKey, dict] = {}
    for episode in episodes:
        for key, diff in _fix_diffs_within(episode.calls).items():
            fix_diffs.setdefault(key, diff)

    return {
        "per_class": _per_class(episodes),
        "api_errors": _error_histogram(episodes, fix_diffs),
        "waste": _waste(episodes),
        "memory_audit": _memory_audit(episodes, memory),
        # Underscore-prefixed and tuple-keyed (unlike "api_errors" above):
        # internal plumbing for build_episode_card, not part of the
        # aggregation table a curator prompt would render. Kept here rather
        # than recomputed per card so every card in an epoch agrees on the
        # same fix — recomputing per call site risks two cards disagreeing
        # about what fixed the same error.
        "_fix_diffs": fix_diffs,
    }
