"""Evaluate the real local skill index without running an agent or an LLM."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from heimdall.skills.registry import Registry, parse_text
from sim.benchmark.env import load_env

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "retrieval_eval/.env"
_PROMPT_SKILL_ID = re.compile(r"^(?P<base>.+)-v[0-9]+(?:-[a-z0-9]+)*$")


def _catalogs(root: Path) -> list[Path]:
    return sorted(path for path in (root, *root.rglob("*")) if any(path.glob("*.md")))


def _cases(root: Path, categories: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for category in categories:
        directory = root / category
        if not directory.is_dir():
            raise ValueError(f"case category does not exist: {directory}")
        for path in sorted(directory.glob("case-*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("status") != "verified":
                continue
            if not isinstance(raw.get("query"), str) or not isinstance(
                raw.get("expected_skills"), list
            ):
                raise ValueError(f"invalid case: {path}")
            rows.append(raw)
    return rows


def _generated_registry(base: Registry, catalog: Path) -> tuple[Registry, dict[str, str]]:
    files = []
    targets = {}
    for path in sorted(catalog.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        skill = parse_text(content, path)
        files.append((path.name, content))
        targets[path.stem] = skill.name
    if not files:
        raise ValueError(f"catalog contains no Markdown skills: {catalog}")
    return base.with_references(files), targets


def _logical_targets(
    registry: Registry, prompt_ids: Iterable[str], exact_targets: set[str],
) -> set[str]:
    families = {
        match.group("base") if (match := _PROMPT_SKILL_ID.fullmatch(prompt_id)) else prompt_id
        for prompt_id in prompt_ids
    }
    return exact_targets | {
        name for name in registry.active() if registry.get(name).name in families
    }


def evaluate(
    *, base_path: Path, generated_root: Path, cases_root: Path,
    categories: Iterable[str], limit: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if limit < 5:
        raise ValueError("limit must be at least 5")
    categories = tuple(categories)
    base = Registry.load(base_path)
    broken = [item for item in base.files() if not item.ok]
    if broken:
        raise ValueError(f"base catalog contains invalid skills: {broken[0].path}")
    cases = _cases(cases_root, categories)
    catalogs = _catalogs(generated_root)
    if not catalogs:
        raise ValueError(f"no generated catalogs found below: {generated_root}")

    details = []
    summaries = []
    uncovered = set()
    for catalog in catalogs:
        registry, targets = _generated_registry(base, catalog)
        catalog_name = catalog.relative_to(generated_root).as_posix()
        rows = []
        for case in cases:
            matched_prompt_ids = sorted(set(case["expected_skills"]) & targets.keys())
            if not matched_prompt_ids:
                uncovered.update(case["expected_skills"])
                continue
            target_names = {targets[prompt_id] for prompt_id in matched_prompt_ids}
            logical_target_names = _logical_targets(
                registry, matched_prompt_ids, target_names,
            )
            result = registry.find(
                case["query"], limit=max(limit, len(registry.active()))
            )["results"]
            ranked = [item for item in result if item["relevance"] > 0]
            target_hits = [
                (rank, item) for rank, item in enumerate(ranked, 1)
                if item["name"] in target_names
            ]
            best_rank, target = min(target_hits, default=(None, None), key=lambda item: item[0])
            logical_rank = next(
                (rank for rank, item in enumerate(ranked, 1)
                 if item["name"] in logical_target_names),
                None,
            )
            row = {
                "catalog": catalog_name,
                "case_id": case["case_id"],
                "category": case["category"],
                "prompt_ids": ",".join(matched_prompt_ids),
                "target_names": ",".join(sorted(target_names)),
                "target_rank": best_rank,
                "target_score": target["relevance"] if target else 0.0,
                "recall_at_1": int(best_rank == 1),
                "recall_at_3": int(best_rank is not None and best_rank <= 3),
                "recall_at_5": int(best_rank is not None and best_rank <= 5),
                "recall_at_10": int(best_rank is not None and best_rank <= 10),
                "reciprocal_rank": 1 / best_rank if best_rank else 0.0,
                "logical_target_rank": logical_rank,
                "logical_recall_at_1": int(logical_rank == 1),
                "logical_recall_at_3": int(logical_rank is not None and logical_rank <= 3),
                "logical_recall_at_5": int(logical_rank is not None and logical_rank <= 5),
                "logical_recall_at_10": int(logical_rank is not None and logical_rank <= 10),
                "logical_reciprocal_rank": 1 / logical_rank if logical_rank else 0.0,
                "zero_score_target": int(best_rank is None),
                "wrong_top_1": int(bool(ranked) and ranked[0]["name"] not in target_names),
                "top_5": json.dumps(
                    [{"name": item["name"], "score": item["relevance"]}
                     for item in ranked[:5]],
                    ensure_ascii=False,
                ),
                "query": case["query"],
            }
            rows.append(row)
            details.append(row)
        if not rows:
            raise ValueError(f"catalog matches none of the selected cases: {catalog}")
        positive_ranks = [row["target_rank"] for row in rows if row["target_rank"]]
        summaries.append({
            "catalog": catalog_name,
            "cases": len(rows),
            "recall_at_1": mean(row["recall_at_1"] for row in rows),
            "recall_at_3": mean(row["recall_at_3"] for row in rows),
            "recall_at_5": mean(row["recall_at_5"] for row in rows),
            "recall_at_10": mean(row["recall_at_10"] for row in rows),
            "mrr": mean(row["reciprocal_rank"] for row in rows),
            "logical_recall_at_1": mean(row["logical_recall_at_1"] for row in rows),
            "logical_recall_at_3": mean(row["logical_recall_at_3"] for row in rows),
            "logical_recall_at_5": mean(row["logical_recall_at_5"] for row in rows),
            "logical_recall_at_10": mean(row["logical_recall_at_10"] for row in rows),
            "logical_mrr": mean(row["logical_reciprocal_rank"] for row in rows),
            "mean_positive_rank": mean(positive_ranks) if positive_ranks else None,
            "zero_score_target_rate": mean(row["zero_score_target"] for row in rows),
            "wrong_top_1_rate": mean(row["wrong_top_1"] for row in rows),
        })
    metadata = {
        "base_catalog": str(base_path.resolve()),
        "generated_root": str(generated_root.resolve()),
        "cases_root": str(cases_root.resolve()),
        "categories": list(categories),
        "source_cases": len(cases),
        "matched_cases_per_catalog": sorted({row["cases"] for row in summaries}),
        "catalogs": len(catalogs),
        "evaluations": len(details),
        "uncovered_expected_skills": sorted(uncovered),
    }
    return details, summaries, metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output: Path, details: list[dict[str, Any]], summaries: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "details.csv", details)
    _write_csv(output / "summary.csv", summaries)
    (output / "summary.json").write_text(
        json.dumps({"metadata": metadata, "catalogs": summaries},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Skill retrieval evaluation", "",
        f"Source cases: {metadata['source_cases']}; matched per catalog: "
        f"{metadata['matched_cases_per_catalog']}; catalogs: {metadata['catalogs']}; "
        f"evaluations: {metadata['evaluations']}.", "",
        "Only targets with `relevance > 0` count as retrieved.", "",
        "Exact means the generated runtime skill; logical also accepts an existing skill "
        "from the same family.", "",
        "| Catalog | Cases | Exact R@1 | Exact R@3 | Exact R@10 | Logical R@1 | Logical R@3 | Logical R@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['catalog']} | {row['cases']} | {row['recall_at_1']:.1%} | "
            f"{row['recall_at_3']:.1%} | {row['recall_at_10']:.1%} | "
            f"{row['logical_recall_at_1']:.1%} | {row['logical_recall_at_3']:.1%} | "
            f"{row['logical_recall_at_10']:.1%} |"
        )
    misses = sorted(
        (row for row in details if not row["recall_at_3"]),
        key=lambda row: (row["catalog"], row["case_id"]),
    )
    lines.extend(["", f"## Recall@3 misses ({len(misses)})", ""])
    lines.extend(
        f"- `{row['catalog']}` / `{row['case_id']}` / `{row['prompt_ids']}`"
        for row in misses
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate Heimdall find_skills offline across generated catalogs."
    )
    base = (os.getenv("HEIMDALL_SKILLS_ROOT") or "heimdall-skills").strip()
    generated = (os.getenv("GENERATED_SKILLS_DIR") or "").strip()
    cases = (os.getenv("CASES") or "").strip()
    result.add_argument("--base", type=Path, default=Path(base))
    result.add_argument("--generated", type=Path, default=Path(generated) if generated else None)
    result.add_argument("--cases", type=Path, default=Path(cases) if cases else None)
    result.add_argument("--category", action="append", dest="categories")
    result.add_argument("--limit", type=int, default=10)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    load_env(ENV_PATH, override=True)
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.generated is None:
        argument_parser.error("set GENERATED_SKILLS_DIR in retrieval_eval/.env or pass --generated")
    if args.cases is None:
        argument_parser.error("set CASES in retrieval_eval/.env or pass --cases")
    base = args.base if args.base.is_absolute() else ROOT / args.base
    generated = args.generated if args.generated.is_absolute() else ROOT / args.generated
    cases = args.cases if args.cases.is_absolute() else ROOT / args.cases
    categories = args.categories or ["answerable"]
    output = args.output or (
        ROOT / "retrieval_eval/results"
        / datetime.now(timezone.utc).strftime("search-%Y%m%dT%H%M%SZ")
    )
    details, summaries, metadata = evaluate(
        base_path=base,
        generated_root=generated,
        cases_root=cases,
        categories=categories,
        limit=args.limit,
    )
    _write_report(output, details, summaries, metadata)
    print(json.dumps({"output": str(output.resolve()), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
