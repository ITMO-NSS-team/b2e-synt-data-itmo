"""Parsing the agent's structured tail.

The parser is deliberately forgiving about everything except the fence: models
drift on spacing, quoting and list punctuation, and a scorer that fails on a
stray space would report an agent error that is really a parser error. What it
is not forgiving about is a missing block — that becomes an explicit
``scored=false`` reason, because a silent zero is indistinguishable from a
wrong answer and would bias whichever arm produces longer output.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.research.answer import ANSWER_CONTRACT, parse_answer   # noqa: E402


def test_parses_a_well_formed_block():
    text = """Разобрал витрину и посчитал.

```answer
verdict: null
ids: []
value: 41
refused: false
reason: null
```"""
    parsed = parse_answer(text)

    assert parsed.present
    assert parsed.value == 41.0
    assert parsed.ids == []
    assert parsed.refused is False
    assert parsed.error is None


def test_parses_an_id_list_with_untidy_punctuation():
    text = """```answer
verdict: null
ids: [ 8f14e45f-ceea-467a-9d0f-2b4c3a1e5b77 , 3c9a1b22-0000-4000-8000-000000000001 ]
value: null
refused: false
reason: null
```"""
    parsed = parse_answer(text)

    assert parsed.ids == ["8f14e45f-ceea-467a-9d0f-2b4c3a1e5b77",
                          "3c9a1b22-0000-4000-8000-000000000001"]


def test_accepts_a_russian_decimal_comma():
    parsed = parse_answer("```answer\nvalue: 0,3125\nrefused: false\n```")

    assert parsed.value == 0.3125


def test_missing_block_is_reported_not_silently_zero():
    parsed = parse_answer("В подразделении 41 сотрудник.")

    assert not parsed.present
    assert parsed.value is None
    assert parsed.error == "no_answer_block"


def test_last_block_wins_when_the_model_emits_two():
    text = "```answer\nvalue: 1\n```\nПоправка.\n```answer\nvalue: 2\n```"

    assert parse_answer(text).value == 2.0


def test_refusal_is_read_even_without_a_value():
    parsed = parse_answer(
        "```answer\nrefused: true\nreason: доступ закрыт\n```")

    assert parsed.refused is True
    assert parsed.reason == "доступ закрыт"


def test_contract_names_every_field_the_parser_reads():
    for key in ("verdict", "ids", "value", "refused", "reason"):
        assert key in ANSWER_CONTRACT


def test_default_system_prompt_carries_the_contract():
    """The prompt and the parser must not drift apart.

    They have no shared type and no import relationship in the running system —
    one is a string sent to the model, the other reads what comes back — so this
    assertion is the only thing keeping them in step.
    """
    from sim.agent.prompt import DEFAULT_SYSTEM_PROMPT

    assert "```answer" in DEFAULT_SYSTEM_PROMPT
    for key in ("verdict", "ids", "value", "refused", "reason"):
        assert key in DEFAULT_SYSTEM_PROMPT
