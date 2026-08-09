"""The judge, and the gate that decides whether it counts.

The judge is the same model family as the agent, so it shares the agent's blind
spots and it rewards length and confidence — both of which a memory arm produces
more of. Two defences are tested here: the payload it sees carries no clue about
which arm produced the answer, and its weight is zero until it demonstrably
agrees with labels that were computed without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.research.judge import (KAPPA_FLOOR, blind, cohen_kappa,      # noqa: E402
                                judge_weight, score_presentation)


def test_blind_strips_every_arm_revealing_field():
    payload = {"question": "q", "answer": "a", "plan": ["mcp_query"],
               "arm": "A3", "memory_block": "...", "config_ref": "agent_config@7",
               "tokens": 31000, "seconds": 84.0}

    out = blind(payload)

    assert set(out) == {"question", "answer", "plan"}
    assert "A3" not in repr(out)


def test_blind_rejects_an_unknown_key_rather_than_passing_it_through():
    """A new field added upstream must fail loudly, not leak silently."""
    with pytest.raises(ValueError, match="memory_digest"):
        blind({"question": "q", "answer": "a", "plan": [], "memory_digest": "abc"})


def test_kappa_is_one_on_perfect_agreement_and_zero_on_chance():
    truth = [True, True, False, False] * 10
    perfect = list(truth)
    chance = [True, False, True, False] * 10

    assert cohen_kappa(perfect, truth) == pytest.approx(1.0)
    assert cohen_kappa(chance, truth) == pytest.approx(0.0, abs=1e-9)


def test_kappa_raises_when_the_calibration_set_has_no_label_variation():
    """A homogeneous truth column makes chance-agreement uncomputable.

    Both raters saying "correct" on every item looks like perfect agreement,
    but with no incorrect examples in the calibration set there is no chance
    model to correct against — a rubber-stamp judge that always says the same
    thing would score kappa=1.0 here by construction, not by having
    demonstrated anything. That must be an error, not a value.
    """
    truth = [True] * 20
    rubber_stamp_judge = [True] * 20

    with pytest.raises(ValueError, match="variation"):
        cohen_kappa(rubber_stamp_judge, truth)


def test_kappa_still_computes_with_only_one_differing_item():
    """One dissenting label is enough for kappa to be well defined."""
    truth = [True] * 19 + [False]
    judge = list(truth)

    assert cohen_kappa(judge, truth) == pytest.approx(1.0)


def test_judge_weight_is_zero_below_the_floor():
    assert judge_weight(KAPPA_FLOOR) == 1.0
    assert judge_weight(KAPPA_FLOOR + 0.1) == 1.0
    assert judge_weight(KAPPA_FLOOR - 0.01) == 0.0


def _client(reply: str):
    """A stand-in for LLMClient.

    ``LLMResponse.text`` is a METHOD on the real type (sim/agent/llm.py:62), not
    an attribute, and ``complete`` takes mandatory ``model`` and ``tools``
    keywords. A fake that got either wrong would let the judge code ship with a
    signature the real client rejects at run time.
    """
    from sim.agent.llm import LLMResponse

    class Fake:
        mode = "fake"
        calls: list[dict] = []

        def complete(self, *, model, system, messages, tools, temperature,
                     max_tokens):
            Fake.calls.append({"model": model, "system": system,
                               "messages": messages, "tools": tools})
            return LLMResponse(content=[{"type": "text", "text": reply}],
                               stop_reason="end_turn", prompt_tokens=1,
                               completion_tokens=1)

    return Fake()


def test_score_presentation_maps_the_rubric_onto_zero_to_one():
    assert score_presentation(_client("score: 4"), question="q",
                              answer="a", plan=[]) == 1.0
    assert score_presentation(_client("score: 2"), question="q",
                              answer="a", plan=[]) == 0.5


def test_unparseable_judge_output_scores_zero_rather_than_guessing():
    assert score_presentation(_client("не могу оценить"), question="q",
                              answer="a", plan=[]) == 0.0


def test_the_judge_is_sent_no_tools_at_all():
    """A judge that could call Heimdall could look the answer up.

    It is a scorer, not an agent; giving it retrieval would let its verdict
    depend on data the answer under review never fetched.
    """
    client = _client("score: 3")
    score_presentation(client, question="q", answer="a", plan=["mcp_query"])

    assert type(client).calls[-1]["tools"] == []
