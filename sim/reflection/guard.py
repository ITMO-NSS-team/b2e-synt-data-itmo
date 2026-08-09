"""Five mechanical rules a proposed lesson must pass before it can be committed.

Any failure discards the text with **no retry**. A retry loop would be the
model searching for phrasing that passes the guard rather than phrasing that
teaches method — which inverts the whole reason a guard exists between the LLM
call and the registry write. ``check`` therefore returns a rule id or
``None``; there is no "try again" return value anywhere in this module.

Why these five and not fewer
-----------------------------
G1/G2 close the same two boundaries ``sim.reflection.extract.query_shape``
closes on the way in — no identifier, no literal absent from evidence — but on
the way *out*: a lesson could still spell out a name or a count in prose even
though the query shape it was mined from never carried one. G3/G4 stop a
lesson from becoming a paraphrase of a basket question (short and generic is
what keeps a lesson from smuggling a question's *content*, long and specific
is what a paraphrase needs room for). G5 stops a lesson from talking about the
scaffolding — tool permissions, refusal policy, the evaluator itself — rather
than the API, which is a different way to leak "what the test wants" without
naming a value at all.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from b2e.gen import dicts
from b2e.gen.names import build_surnames

# ------------------------------------------------------------------ G1: PII

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")

#: Every surname form the corpus's name generator can produce (see
#: ``b2e/gen/names.py``), flattened from (masculine, feminine) pairs. Built
#: once at import time — 474 pairs, cheap — rather than per call.
_SURNAMES = frozenset(
    form for pair in build_surnames() for form in pair)

#: Fixed vocabulary org unit names are composed from (``b2e/gen/org.py``
#: builds every unit name as one of these words, or one of these words plus a
#: topic). A unit's exact leaf name depends on the snapshot's seed and cannot
#: be enumerated in advance, but the *words that mean "this is a unit name"*
#: are a closed, seed-independent set — level 1/2 names are literal entries of
#: ``BLOCKS``/``TERRITORIAL_BANKS``, and every level 3-5 name starts with one
#: of ``UNIT_KINDS``. Matching on that fixed vocabulary is real signal, not a
#: guess: an ordinary API-mechanics lesson has no reason to say "Департамент"
#: or "Розничный бизнес" at all.
def _stem(word: str) -> str:
    """Strip a plausible Russian case ending so "Управлении" (locative) still
    matches the dictionary's nominative "Управление". Crude on purpose: this
    is a guard against an org unit name leaking into agent-facing text, not a
    morphological analyser, and a stem that is a few characters too short
    only widens the match, never narrows it past the words this function is
    built from."""
    if len(word) >= 6:
        return word[:-2]
    if len(word) >= 4:
        return word[:-1]
    return word


_UNIT_NAME_PHRASES: tuple[str, ...] = (
    tuple(name for name, _weight in dicts.BLOCKS)
    + tuple(name for name, _weight in dicts.TERRITORIAL_BANKS)
    + tuple(dicts.UNIT_KINDS)
)
#: Stems of every significant (4+ letter) word across the org-name vocabulary
#: above, so a declined form of a unit name still matches. Deliberately flat
#: across phrases rather than requiring a whole phrase: a lesson naming just
#: the block ("Розничный") or just the unit kind ("Управление") is already
#: naming an organisational identifier, not describing API mechanics.
_UNIT_NAME_STEMS: frozenset[str] = frozenset(
    _stem(word) for phrase in _UNIT_NAME_PHRASES
    for word in re.findall(r"[А-ЯЁа-яё]+", phrase) if len(word) >= 4
)
_UNIT_NAME_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(s) for s in
                             sorted(_UNIT_NAME_STEMS, key=len, reverse=True))
    + r")\w*")


def _rule_g1(text: str) -> bool:
    """True if ``text`` names an identifier a lesson must never carry."""
    if _UUID_RE.search(text):
        return True
    words = re.findall(r"[А-ЯЁ][а-яё]+", text)
    if any(w in _SURNAMES for w in words):
        return True
    if _UNIT_NAME_RE.search(text):
        return True
    return False


# --------------------------------------------------------- G2: numeric literals

#: A number as it appears in Russian prose: digit groups optionally separated
#: by thin/regular spaces (thousands), with an optional decimal part.
_NUMBER_RE = re.compile(r"\d[\d  ]*(?:[.,]\d+)?")


def _significant_digits(token: str) -> str:
    return re.sub(r"[^\d]", "", token)


def _evidence_numbers(evidence: Any) -> frozenset[str]:
    """Every numeric leaf reachable from ``evidence``, as bare digit strings.

    Walks dicts, lists and tuples rather than assuming a fixed shape: the
    aggregation table Task 4's own ``aggregate()`` produces nests per-class
    stats, error histograms and waste counts several levels deep, and a flat
    ``evidence.keys()`` scan would miss all of it. Dict *keys* are scanned too
    — ``aggregate()``'s own error histogram encodes an HTTP status into its
    key string (``"mcp_query|403|forbidden|..."``), and a lesson citing that
    status is citing real evidence even though it never appears as a value.
    """
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            out.add(_significant_digits(str(node)))
        elif isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str):
                    for match in _NUMBER_RE.finditer(k):
                        out.add(_significant_digits(match.group()))
                walk(v)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            for match in _NUMBER_RE.finditer(node):
                out.add(_significant_digits(match.group()))

    walk(evidence)
    out.discard("")
    return frozenset(out)


def _rule_g2(text: str, evidence: Any) -> bool:
    """True if ``text`` states a number the evidence cannot back up.

    ``1-99`` survive unconditionally — "не более 2 запросов" is method, not a
    fact that needs sourcing — because a count that small is describing a
    *shape* (how many calls, how many retries), not asserting a *measurement*
    a model could only know from having peeked at the answer key.
    """
    allowed = _evidence_numbers(evidence)
    for match in _NUMBER_RE.finditer(text):
        digits = _significant_digits(match.group())
        if not digits:
            continue
        if len(digits) < 3:
            continue
        try:
            magnitude = int(digits)
        except ValueError:
            magnitude = None
        if magnitude is not None and 1 <= magnitude <= 99:
            continue
        if digits not in allowed:
            return True
    return False


# ------------------------------------------------------------------- G3: length

_SENTENCE_RE = re.compile(r"[.!?]+")


def _rule_g3(text: str) -> bool:
    if len(text) > 240:
        return True
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    return len(sentences) > 2


# ------------------------------------------------------ G4: question paraphrase

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_JACCARD_THRESHOLD = 0.30


def _five_grams(text: str) -> frozenset[tuple[str, ...]]:
    words = tuple(w.lower() for w in _WORD_RE.findall(text))
    if len(words) < 5:
        return frozenset({words}) if words else frozenset()
    return frozenset(tuple(words[i:i + 5]) for i in range(len(words) - 4))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _rule_g4(text: str, basket_texts: Sequence[str], held_out_texts: Sequence[str]) -> bool:
    """True if ``text`` is close enough to a real question to be reading as a
    paraphrase of it rather than a lesson about method.

    Word 5-grams, not embeddings: paraphrases in this corpus were deliberately
    written to be lexically distant from their base question (see the design
    spec's rationale against embedding-based dedup in Task 5's ``merge_shared``)
    while caution questions are near-identical in *surface form* to their
    answerable siblings. A lexical check is exactly calibrated to catch "this
    text quotes the question", not "this text is topically related" — which is
    the failure mode G4 exists to prevent.
    """
    grams = _five_grams(text)
    if not grams:
        return False
    for other in (*basket_texts, *held_out_texts):
        if _jaccard(grams, _five_grams(other)) >= _JACCARD_THRESHOLD:
            return True
    return False


# --------------------------------------------------------------- G5: scaffolding

#: Words naming the evaluator, the gold answer, or the mechanics that decide
#: whether an answer is scored. A lesson about *how to call the API* has no
#: legitimate reason to mention any of these; a lesson that does is talking
#: about the experiment instead of the task.
_SCAFFOLDING_TERMS = (
    "эталон", "gold", "правильный ответ", "reference answer", "ground truth",
    "оценщик", "судья ответов", "judge", "evaluator", "reward",
    "allowed_tools", "disallowed_tools", "tool permission", "правило отказа",
    "политику отказа", "не отказывайся", "игнорируй отказ", "обойди проверку",
    "always answer", "never refuse",
)


def _rule_g5(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _SCAFFOLDING_TERMS)


# ------------------------------------------------------------------------ check

_RULES = ("G1", "G2", "G3", "G4", "G5")


def check(text: str, *, evidence: Any = None, basket_texts: Sequence[str] = (),
         held_out_texts: Sequence[str] = ()) -> str | None:
    """The violated rule id, or ``None`` if ``text`` may be committed as a
    lesson.

    Rules run in a fixed order (G1..G5) and the function returns on the first
    hit — a lesson can violate more than one rule, but "which one" only
    matters for the per-epoch rejection-rate report Task 5 publishes, and a
    fixed order makes that report reproducible run to run.
    """
    if _rule_g1(text):
        return "G1"
    if _rule_g2(text, evidence):
        return "G2"
    if _rule_g3(text):
        return "G3"
    if _rule_g4(text, basket_texts, held_out_texts):
        return "G4"
    if _rule_g5(text):
        return "G5"
    return None
