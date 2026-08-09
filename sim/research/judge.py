"""The presentation rubric, and the gate that decides whether it counts.

The judge is worth 15 of 100 points and nothing else. That ceiling is
deliberate: the obvious judge is the same Haiku 4.5 the agent runs on, so it
shares the agent's blind spots, and it is known to reward length, structure and
confident phrasing — all of which a memory block reliably increases. A judge
carrying the headline would let a style gain be reported as a quality gain.

Two mechanical defences. ``blind`` removes every field that could reveal which
arm produced an answer and *raises* on any key it does not recognise, so a field
added upstream fails loudly instead of leaking. And ``judge_weight`` returns
zero until the judge agrees with deterministic labels at kappa >= 0.60, a
threshold fixed here rather than chosen after the effect size is known.
"""
from __future__ import annotations

import re

#: The only keys a judge may see. Everything else — arm, memory block, config
#: ref, token counts, timings — is a channel through which the treatment could
#: reach the scorer.
_ALLOWED = frozenset({"question", "answer", "plan"})

KAPPA_FLOOR = 0.60

JUDGE_PROMPT = """Оцени качество изложения ответа кадрового ассистента по шкале 0-4.

Ты НЕ проверяешь фактическую правильность — она уже проверена отдельно и
машинально. Оценивай только изложение:

4 — каждое утверждение опирается на полученные данные, структура ясна,
    неопределённость названа явно
3 — в основном обосновано, мелкие огрехи структуры
2 — есть необоснованные утверждения либо изложение путаное
1 — утверждения в основном не подкреплены
0 — бессвязно либо ответ отсутствует

Ответь ровно одной строкой вида: score: N
"""

_SCORE = re.compile(r"score\s*:\s*([0-4])", re.I)


def blind(payload: dict) -> dict:
    """Strip everything that could tell the judge which arm it is scoring."""
    unknown = set(payload) - _ALLOWED - {
        "arm", "memory_block", "config_ref", "tokens", "seconds"}
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in judge payload: add them to the "
            "allow-list or the strip-list deliberately, never by default")
    return {k: v for k, v in payload.items() if k in _ALLOWED}


#: The judge runs on the same cheap model as the agent. That is a known bias —
#: shared blind spots — and it is why the judge is capped at 15 of 100 points
#: and gated on kappa rather than trusted.
JUDGE_MODEL = "claude-haiku-4-5-20251001"


def score_presentation(client, *, question: str, answer: str,
                       plan: list[str]) -> float:
    """The rubric score on 0-1, or 0.0 if the judge did not answer in shape.

    Zero rather than a retry: a retry loop is the scorer being asked again until
    it produces a parseable number, which biases toward whatever the model says
    most readily.

    ``tools=[]`` is not an omission. A judge that could query Heimdall could look
    the answer up, and its verdict would then depend on data the answer under
    review never fetched.
    """
    payload = blind({"question": question, "answer": answer, "plan": plan})
    body = (f"Вопрос: {payload['question']}\n\n"
            f"План обращений к API: {payload['plan']}\n\n"
            f"Ответ: {payload['answer']}")
    response = client.complete(
        model=JUDGE_MODEL, system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": body}],
        tools=[], temperature=0.0, max_tokens=16)
    # LLMResponse.text is a method (sim/agent/llm.py:62), not an attribute.
    found = _SCORE.search(response.text() or "")
    return (int(found.group(1)) / 4.0) if found else 0.0


def cohen_kappa(judge: list[bool], truth: list[bool]) -> float:
    """Agreement beyond chance between the judge and the deterministic label."""
    n = len(truth)
    if n == 0 or len(judge) != n:
        raise ValueError("judge and truth must be non-empty and the same length")
    observed = sum(1 for a, b in zip(judge, truth) if a == b) / n
    pj, pt = sum(judge) / n, sum(truth) / n
    expected = pj * pt + (1 - pj) * (1 - pt)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def judge_weight(kappa: float) -> float:
    """1.0 if the judge has earned its 15 points, 0.0 otherwise.

    Binary rather than a smooth taper, so the decision is a pre-registered
    threshold rather than a dial that can be nudged once the arms are in.
    """
    return 1.0 if kappa >= KAPPA_FLOOR else 0.0
