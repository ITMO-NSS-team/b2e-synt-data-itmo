from __future__ import annotations

import json
from pathlib import Path

from scripts.judge_existing_results import run_posthoc


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeJudge:
    model = "fake-judge"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, **kwargs):
        self.calls += 1
        return {
            "applicable": True,
            "pass": "good" in kwargs["candidate"],
            "scores": {"semantic_correctness": int("good" in kwargs["candidate"])},
            "explanation": "checked",
            "model": self.model,
            "prompt_version": "test",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "error": None,
        }


def test_posthoc_judges_saved_answers_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "judge"
    source.mkdir()
    responses = []
    scores = []
    for index, answer in enumerate(("good answer", "bad answer"), 1):
        run_id = f"run-{index}"
        responses.append({
            "run_id": run_id,
            "case_id": f"case-{index}",
            "mode": "mode-a",
            "repetition": 1,
            "status": "completed",
            "case_snapshot": {
                "category": "answerable",
                "original_query": "query",
                "gold_answer": {"outcome": "answer", "rows": []},
                "gold_comparison": {},
            },
            "response": {"raw_answer": answer},
        })
        scores.append({
            "run_id": run_id,
            "metrics": {"llm_judge_accuracy": None},
            "llm_judge": {"applicable": False},
        })
    _write_jsonl(source / "responses.jsonl", responses)
    _write_jsonl(source / "scores.jsonl", scores)

    judge = FakeJudge()
    summary = run_posthoc(source, output, judge, workers=2, retry_delay=0)
    assert summary == {
        "runs": 2,
        "scored": 2,
        "passed": 1,
        "errors": 0,
        "accuracy": 0.5,
        "by_mode": {
            "mode-a": {"runs": 2, "scored": 2, "passed": 1, "errors": 0, "accuracy": 0.5},
        },
    }
    assert judge.calls == 2
    augmented = [json.loads(line) for line in (
        output / "scores-with-llm-judge.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert [row["metrics"]["llm_judge_accuracy"] for row in augmented] == [1, 0]

    run_posthoc(source, output, judge, workers=2, retry_delay=0)
    assert judge.calls == 2
    assert json.loads((output / "manifest.json").read_text())["workers"] == 2
