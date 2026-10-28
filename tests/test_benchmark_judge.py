from __future__ import annotations

import json

import httpx

from sim.benchmark.judge import ZaiAnswerJudge, from_env


def test_zai_judge_sends_only_answer_material_and_parses_verdict() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "pass": True, "explanation": "Смысл совпадает",
            }, ensure_ascii=False)}}],
            "usage": {
                "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25,
            },
        })

    result = ZaiAnswerJudge(
        model="glm-test", api_key="secret", base_url="https://z.ai/v4",
        transport=httpx.MockTransport(handler),
    ).evaluate(
        query="Кто работает в команде?",
        reference={"outcome": "answer", "rows": [{"name": "Иван"}]},
        comparison={"ordered": False},
        candidate="В команде работает Иван.",
    )

    user = json.loads(captured["messages"][1]["content"])
    assert result["pass"] is True
    assert result["usage"]["total_tokens"] == 25
    assert captured["model"] == "glm-test"
    assert user["candidate_answer"] == "В команде работает Иван."
    assert "trace" not in captured["messages"][1]["content"]
    assert "skill" not in captured["messages"][1]["content"]
    assert "mode" not in captured["messages"][1]["content"]


def test_judge_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("BENCH_LLM_JUDGE_ENABLED", raising=False)
    assert from_env() is None


def test_judge_uses_its_own_model(monkeypatch) -> None:
    monkeypatch.setenv("BENCH_LLM_JUDGE_ENABLED", "true")
    monkeypatch.setenv("BENCH_LLM_JUDGE_MODEL", "glm-judge")
    monkeypatch.setenv("B2E_MODEL", "glm-agent")
    monkeypatch.setenv("ZAI_API_KEY", "secret")
    monkeypatch.setenv("ZAI_OPENAI_BASE_URL", "https://z.ai/v4")

    judge = from_env()

    assert isinstance(judge, ZaiAnswerJudge)
    assert judge.model == "glm-judge"
