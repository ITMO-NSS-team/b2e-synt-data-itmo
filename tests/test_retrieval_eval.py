from __future__ import annotations

import json
from pathlib import Path

from retrieval_eval.__main__ import evaluate


def skill(name: str, description: str) -> str:
    return (
        f"---\nname: {name}\ntitle: {description}\nkind: reference\n"
        f"domain: general\ndescription: {description}\n---\n"
    )


def test_evaluate_counts_only_positive_score_targets(tmp_path: Path) -> None:
    base = tmp_path / "base"
    generated = tmp_path / "generated" / "compact" / "detailed"
    cases = tmp_path / "cases" / "answerable"
    base.mkdir()
    generated.mkdir(parents=True)
    cases.mkdir(parents=True)
    (base / "candidate_search.md").write_text(
        skill("candidate_search", "найди кандидатов по навыкам"), encoding="utf-8"
    )
    (generated / "candidate_search-v1.md").write_text(
        skill("candidate-search-generated", "поиск кандидатов с навыками"), encoding="utf-8"
    )
    for case_id, query in (("case-1", "найди кандидатов по навыкам"),
                           ("case-2", "совершенно посторонний запрос")):
        (cases / f"{case_id}.json").write_text(json.dumps({
            "case_id": case_id,
            "category": "answerable",
            "status": "verified",
            "query": query,
            "expected_skills": ["candidate_search-v1"],
        }, ensure_ascii=False), encoding="utf-8")

    details, summaries, metadata = evaluate(
        base_path=base,
        generated_root=tmp_path / "generated",
        cases_root=tmp_path / "cases",
        categories=["answerable"],
    )

    assert metadata["evaluations"] == 2
    assert [row["target_rank"] for row in details] == [2, None]
    assert [row["logical_target_rank"] for row in details] == [1, None]
    assert summaries[0]["recall_at_1"] == 0
    assert summaries[0]["logical_recall_at_1"] == 0.5
    assert summaries[0]["zero_score_target_rate"] == 0.5
