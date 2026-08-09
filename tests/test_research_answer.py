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


# --------------------------------------------- fix round 1: adversarial inputs


def test_parses_a_yaml_block_list_for_ids():
    """The contract shows flow style (``ids: [a, b]``), but a block list —
    ``ids:`` followed by indented ``- item`` lines — is just as common a model
    default. Losing it to ``[]`` is the same defect as losing a percent-suffixed
    number: a shape the model chose in good faith reads as "no answer"."""
    text = """```answer
verdict: null
ids:
  - 8f14e45f-ceea-467a-9d0f-2b4c3a1e5b77
  - 3c9a1b22-0000-4000-8000-000000000001
value: null
refused: false
reason: null
```"""
    parsed = parse_answer(text)

    assert parsed.ids == ["8f14e45f-ceea-467a-9d0f-2b4c3a1e5b77",
                          "3c9a1b22-0000-4000-8000-000000000001"]
    assert "ids" not in parsed.field_errors


def test_percent_suffixed_value_is_read_as_a_fraction():
    """"%" is division by 100 by definition, and every share-shaped question
    class (``ReferenceSpec(op="share", ...)`` in ``sim/oracle/reference.py``)
    expects a fraction of one — the same convention the Russian-decimal-comma
    test already exercises for 0.3125. So 42% reads as 0.42, not 42."""
    parsed = parse_answer("```answer\nvalue: 42%\n```")

    assert parsed.value == 0.42
    assert "value" not in parsed.field_errors


def test_unparseable_value_is_flagged_not_silently_none():
    """A fraction like 1/3 is not a shape the parser reads. The failure must
    be visible on the object, not collapse into the same ``None`` an explicit
    ``value: null`` produces — those are different situations for a scorer."""
    parsed = parse_answer("```answer\nvalue: 1/3\n```")

    assert parsed.present
    assert parsed.value is None
    assert parsed.field_errors.get("value") is not None


def test_three_states_are_distinguishable_for_the_same_field():
    """absent, present-and-read, present-and-unreadable must all be tellable
    apart for a given field. Absent and an explicit null are the same "no
    answer" state; only genuinely unreadable content sets a field error."""
    absent = parse_answer("```answer\nrefused: false\n```")
    assert absent.value is None
    assert "value" not in absent.field_errors

    explicit_null = parse_answer("```answer\nvalue: null\n```")
    assert explicit_null.value is None
    assert "value" not in explicit_null.field_errors

    read = parse_answer("```answer\nvalue: 41\n```")
    assert read.value == 41.0
    assert "value" not in read.field_errors

    unreadable = parse_answer("```answer\nvalue: 1/3\n```")
    assert unreadable.value is None
    assert "value" in unreadable.field_errors


def test_unparseable_refused_is_flagged_too():
    """The same defect applies to ``refused``: an unrecognised spelling must
    not silently read as False, which is what "not refused" also looks like."""
    parsed = parse_answer("```answer\nrefused: maybe\n```")

    assert parsed.refused is False
    assert "refused" in parsed.field_errors
