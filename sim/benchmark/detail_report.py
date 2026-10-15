"""Paired detail analysis and local HTML/CSV/JSON report; no LLM calls."""
from __future__ import annotations
import argparse
import csv
import html
import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from .env import load_env
from .execution import OPERATIONAL_METRICS
from .results import _metric_means

SCORED = {"completed", "normalization_pending"}
METRICS = ("answer_accuracy", "exact_match", "outcome_accuracy", "correct_refusal", "generated_skill_loaded", *OPERATIONAL_METRICS)
LIMITATIONS = [
    "Один generated-артефакт на метод/детализацию/навык: случайность генерации не реплицирована.",
    "Проверяются готовые описания разной детализации; интерактивный этап уточнения SF напрямую не проверяется.",
    "Аугментации кейсов зависимы. Интервалы получены bootstrap по типам навыков, а не по независимым строкам кейсов.",
    "Ошибки формата считаются неверными ответами. Инфраструктурные сбои исключены из условной accuracy; coverage и успех на запланированный запуск показаны отдельно.",
    "Методы анализируются раздельно. Порядок фаз фиксируется seed; изменения провайдера и прогрев кэша могут влиять на время и стоимость.",
    "Задержки измеряются при параллельных сессиях и не являются оценкой задержки одиночного запроса.",
    "Стоимость генерации и уточнения отсутствует; стоимость в отчёте относится только к выполнению агентом.",
]


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def _skill(plan, case_id):
    return "+".join(sorted(plan["case_skills"][case_id]))


def _aggregate(rows):
    scored = [row for row in rows if row["status"] in SCORED and row.get("metrics")]
    cost = [row["metrics"]["cost_usd"] for row in rows if row["metrics"].get("cost_usd") is not None]
    return {
        "n_runs": len(rows), "n_scored": len(scored),
        "statuses": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
        "score_coverage": len(scored) / len(rows) if rows else None,
        "success_per_scheduled_run": sum(row["metrics"].get("answer_accuracy") or 0 for row in scored) / len(rows) if rows else None,
        **_metric_means(row["metrics"] for row in scored),
        "cost_usd_total_known": sum(cost) if cost else None, "n_cost_measured": len(cost),
    }


def _paired(plan, pairs):
    deltas = {}
    for metric in METRICS:
        values = [(a["metrics"].get(metric), b["metrics"].get(metric)) for a, b in pairs if a["metrics"].get(metric) is not None and b["metrics"].get(metric) is not None]
        deltas[metric] = sum(a - b for a, b in values) / len(values) if values else None
    clusters = defaultdict(list)
    for a, b in pairs:
        clusters[_skill(plan, a["case_id"])].append(a["metrics"]["answer_accuracy"] - b["metrics"]["answer_accuracy"])
    means = [sum(values) / len(values) for values in clusters.values()]
    ci = None
    if len(means) >= 2:
        rng = random.Random(271828)
        samples = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(2000))
        ci = [100 * samples[49], 100 * samples[1949]]
    return {
        "n_pairs": len(pairs), "n_skill_clusters": len(means),
        "accuracy_delta_pp": 100 * deltas["answer_accuracy"] if pairs else None,
        "macro_skill_accuracy_delta_pp": 100 * sum(means) / len(means) if means else None,
        "macro_skill_delta_ci95_pp": ci,
        "wins": sum(a["metrics"]["answer_accuracy"] > b["metrics"]["answer_accuracy"] for a, b in pairs),
        "losses": sum(a["metrics"]["answer_accuracy"] < b["metrics"]["answer_accuracy"] for a, b in pairs),
        "ties": sum(a["metrics"]["answer_accuracy"] == b["metrics"]["answer_accuracy"] for a, b in pairs),
        "metric_deltas": deltas,
    }


def analyze_detail(plan, responses, scores, *, repetitions=1):
    score_map = {row["run_id"]: row for row in scores}
    groups = {"generated_skills@" + group["variant_id"]: group for group in plan["groups"]}
    keyed, arms = {}, defaultdict(list)
    for raw in responses:
        key = (raw["mode"], raw["case_id"], int(raw["repetition"]))
        if key in keyed:
            raise ValueError(f"duplicate report cell: {key}")
        row = {**raw, "metrics": score_map.get(raw["run_id"], {}).get("metrics", {})}
        keyed[key] = row
        arms[raw["mode"]].append(row)
    by_detail, baselines, comparisons, by_skill = {}, {}, {}, {}
    for mode, rows in arms.items():
        group = groups.get(mode)
        if group:
            by_detail.setdefault(group["generation_method"], {})[group["detail_level"]] = _aggregate(rows)
        else:
            baselines[mode] = _aggregate(rows)
    for method in plan["methods"]:
        modes = {group["detail_level"]: "generated_skills@" + group["variant_id"] for group in plan["groups"] if group["generation_method"] == method}
        by_skill[method], comparisons[method] = {}, {}
        for level, mode in modes.items():
            buckets = defaultdict(list)
            for row in arms.get(mode, []):
                buckets[_skill(plan, row["case_id"])].append(row)
            by_skill[method][level] = {name: _aggregate(rows) for name, rows in sorted(buckets.items())}
            skill_accuracy = [stats["answer_accuracy"] for stats in by_skill[method][level].values() if stats["answer_accuracy"] is not None]
            if level in by_detail.get(method, {}):
                by_detail[method][level]["macro_skill_accuracy"] = sum(skill_accuracy) / len(skill_accuracy) if skill_accuracy else None
        for baseline, target in itertools.combinations(plan["detail_levels"], 2):
            cells = sorted({(case, repeat) for mode, case, repeat in keyed if mode in {modes[target], modes[baseline]}})
            pairs = []
            for case, repeat in cells:
                a, b = keyed.get((modes[target], case, repeat)), keyed.get((modes[baseline], case, repeat))
                if a and b and a["status"] in SCORED and b["status"] in SCORED and a["metrics"].get("answer_accuracy") is not None and b["metrics"].get("answer_accuracy") is not None:
                    pairs.append((a, b))
            comparisons[method][f"{target}_vs_{baseline}"] = _paired(plan, pairs)
    expected = len(plan["selected_case_ids"]) * repetitions
    return {
        "schema_version": "1.0", "experiment": "prompt_detail", "n_responses": len(responses),
        "expected_runs_per_arm": expected, "methods": plan["methods"], "detail_levels": plan["detail_levels"],
        "n_source_cases": len(plan["source_case_ids"]), "n_selected_cases": len(plan["selected_case_ids"]),
        "n_excluded_cases": len(plan["excluded"]), "excluded": plan["excluded"],
        "n_skills": len({name for names in plan["case_skills"].values() for name in names}),
        "n_catalog_skills": len(plan["common_skills"]), "repetitions": repetitions,
        "case_selection": {**plan.get("case_selection", {"strategy": "all_verified"}), "case_limit": plan.get("request", {}).get("limit")},
        "by_detail": by_detail, "baselines": baselines, "paired_comparisons": comparisons, "by_skill": by_skill,
        "complete_generated_matrix": all(len(arms.get(mode, [])) == expected for mode in groups),
        "limitations": LIMITATIONS,
    }


def _format(value, percent=False):
    return "—" if value is None else (f"{100 * value:.1f}%" if percent else f"{value:.2f}")


def _table(report):
    result = []
    for method in report["methods"]:
        for level in report["detail_levels"]:
            stats = report["by_detail"].get(method, {}).get(level, {})
            result.append([method, level, f"{stats.get('n_scored', 0)}/{stats.get('n_runs', 0)}",
                _format(stats.get("answer_accuracy"), True), _format(stats.get("macro_skill_accuracy"), True), _format(stats.get("score_coverage"), True),
                _format(stats.get("generated_skill_loaded"), True), _format(stats.get("mcp_query_calls")),
                _format(stats.get("failed_tool_calls")), _format(stats.get("total_tokens")),
                _format(stats["latency_ms"] / 1000 if stats.get("latency_ms") is not None else None),
                _format(stats.get("cost_usd_total_known"))])
    return result


def write_detail_report(result_dir, *, phoenix_url="", project_id=""):
    root = Path(result_dir).resolve()
    manifest = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    plan = manifest["generated_plan"]
    if plan.get("experiment") != "prompt_detail":
        raise ValueError("detail experiment manifest required")
    modes = manifest["modes"]
    if not modes or any(mode.get("common") != modes[0].get("common") for mode in modes):
        raise ValueError("model/prompt/snapshot/access conditions differ across arms")
    for phase in manifest.get("catalog_phases", []):
        if phase["live_stand"].get("skill_registry_hash") != manifest["live_stand"].get("skill_registry_hash"):
            raise ValueError("executable registry changed between phases")
    responses, scores = _read_lines(root / "responses.jsonl"), _read_lines(root / "scores.jsonl")
    report = analyze_detail(plan, responses, scores, repetitions=int(manifest["repetitions"]))
    report.update({"eval_id": manifest["eval_id"], "conditions": modes[0]["common"],
        "workers": manifest.get("workers", 1), "phoenix_url": phoenix_url,
        "phase_order": [group["variant_id"] for group in plan["groups"]],
        "baseline_comparisons": json.loads((root / "summary.json").read_text(encoding="utf-8")).get("comparisons", {})})
    brief = ["# Краткая сводка", "", f"Эксперимент `{report['eval_id']}`: {report['n_selected_cases']} кейсов, {report['n_skills']} типов навыков, {report['repetitions']} повторов.", ""]
    for method in report["methods"]:
        comparisons = report["paired_comparisons"][method]
        detailed = comparisons.get("detailed_vs_abstract", {})
        ci = detailed.get("macro_skill_delta_ci95_pp")
        delta = detailed.get("accuracy_delta_pp")
        n_pairs = detailed.get("n_pairs", 0)
        if not n_pairs:
            verdict = "нет общих оценённых пар для сравнения detailed с abstract"
        elif ci is None:
            verdict = "слишком мало типов навыков для интервала; результат описательный"
        elif ci[0] > 0:
            verdict = "detailed показывает преимущество по средней точности с равным весом навыков в этой выборке"
        elif ci[1] < 0:
            verdict = "detailed показывает ухудшение по средней точности с равным весом навыков в этой выборке"
        else:
            verdict = "устойчивое преимущество detailed по типам навыков не подтверждено: интервал включает ноль"
        brief.append(f"- {method}: {verdict}; {n_pairs} пар, Δ accuracy по кейсам {_format(delta)} п.п., 95% cluster CI macro-дельты {ci}.")
    brief += ["", "Это сравнение готовых артефактов из описаний разной детализации. Оно не является прямым экспериментом с интерактивным уточнением SF и само по себе не доказывает необходимость или ненужность этого этапа.", "", "Пропуски и инфраструктурные ошибки смотрите в coverage/statuses. Generation cost и clarification cost неизвестны. Доступные метрики выполнения, каждый ответ и трейсы сохранены в полном отчёте и CSV."]
    (root / "brief-summary.md").write_text("\n".join(brief) + "\n", encoding="utf-8")
    (root / "detail-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_map = {row["run_id"]: row for row in scores}
    groups = {"generated_skills@" + group["variant_id"]: group for group in plan["groups"]}
    fields = ["case_id", "canonical_skill", "generation_method", "detail_level", "repetition", "mode", "status", *METRICS, "session_id", "trace_id", "trace_file", "trace_url"]
    with (root / "detail-runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in responses:
            group, trace_id = groups.get(raw["mode"], {}), raw.get("trace_id")
            url = f"{phoenix_url.rstrip('/')}/projects/{quote(project_id, safe='')}/traces/{quote(trace_id, safe='')}" if trace_id and project_id and phoenix_url else phoenix_url
            writer.writerow({**{key: raw.get(key) for key in ("case_id", "repetition", "mode", "status", "session_id", "trace_id")},
                **{key: score_map.get(raw["run_id"], {}).get("metrics", {}).get(key) for key in METRICS},
                "canonical_skill": _skill(plan, raw["case_id"]), "generation_method": group.get("generation_method"),
                "detail_level": group.get("detail_level"), "trace_file": raw.get("trace_path"), "trace_url": url})
    headers = ["Метод", "Детализация", "Оценено/запущено", "Accuracy", "Macro accuracy", "Coverage", "Скил загружен", "MCP", "Ошибки tools", "Токены", "Секунды", "Измеренная стоимость, $"]
    table = _table(report)
    if any(len(row) != len(headers) for row in table):
        raise ValueError("report table columns do not match headers")
    lines = ["# Влияние детализации описания на качество скилов", "", f"Эксперимент `{report['eval_id']}`: {report['n_selected_cases']} кейсов, {report['n_skills']} типов навыков, {report['repetitions']} повторов, {report['workers']} параллельных сессий.",
        f"Модель `{report['conditions']['model_id']}`, harness `{report['conditions']['harness']}`.", "",
        "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in table]
    lines += ["", "Выбор кейсов: " + json.dumps(report["case_selection"], ensure_ascii=False) + "."]
    lines += ["", "## Парные сравнения", ""]
    comparison_rows = []
    for method, pairs in report["paired_comparisons"].items():
        for pair, stats in pairs.items():
            row = [method, pair, str(stats["n_pairs"]), _format(stats["accuracy_delta_pp"]), _format(stats["macro_skill_accuracy_delta_pp"]), str(stats["macro_skill_delta_ci95_pp"])]
            comparison_rows.append(row)
            lines.append(f"- {method}, {pair}: {stats['n_pairs']} пар; Δ accuracy {row[3]} п.п.; Δ с равным весом навыков {row[4]} п.п.; 95% cluster CI {row[5]}; улучшений/ухудшений/ничьих {stats['wins']}/{stats['losses']}/{stats['ties']}.")
    lines += ["", "## Где смотреть", "", f"- Phoenix: {phoenix_url}, проект `b2e-itmo`, metadata `eval_id={report['eval_id']}`.",
        "- `detail-runs.csv`: метрики каждого запуска, trace_id и путь к трейсу.", "- `detail-report.json`: разбивка по навыкам, coverage, статусы и парные сравнения.",
        "- `responses.jsonl`, `scores.jsonl`, `traces/`: исходные ответы, нормализованные результаты и полные трассы.", "", "## Ограничения", ""]
    lines += ["- " + item for item in LIMITATIONS]
    (root / "detail-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    def cells(rows):
        return "".join("<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in row) + "</tr>" for row in rows)
    document = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Skill Factory: детализация</title>
<style>body{{font:15px system-ui;margin:32px;color:#182033}}table{{border-collapse:collapse;width:100%;margin:24px 0}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#f2f5fa}}a{{color:#2563eb}}li{{margin:8px 0}}</style>
<h1>Влияние детализации описания на качество скилов</h1><p>{html.escape(lines[2])}</p><p>{html.escape(lines[3])}</p>
<table><tr>{''.join('<th>'+html.escape(value)+'</th>' for value in headers)}</tr>{cells(table)}</table>
<h2>Парные изменения в процентных пунктах</h2><table><tr><th>Метод</th><th>Сравнение</th><th>Пар</th><th>Δ accuracy</th><th>Δ по навыкам</th><th>95% cluster CI</th></tr>{cells(comparison_rows)}</table>
<p>Методы генерации не объединяются. Пропуски из-за инфраструктуры показаны отдельно. Интерактивное уточнение SF напрямую не проверяется.</p>
<p><a href="detail-runs.csv">Запуски и trace_id (CSV)</a> · <a href="detail-report.json">Полные метрики (JSON)</a> · <a href="detail-report.md">Текстовый отчёт</a> · <a href="{html.escape(phoenix_url, quote=True)}">Phoenix</a></p>
<h2>Ограничения</h2><ul>{''.join('<li>'+html.escape(item)+'</li>' for item in LIMITATIONS)}</ul></html>"""
    (root / "detail-report.html").write_text(document, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--phoenix-url")
    parser.add_argument("--phoenix-project-id", default="")
    args = parser.parse_args(argv)
    load_env()
    url = args.phoenix_url or os.getenv("PUBLIC_URL", "https://localhost:8443").rstrip("/") + "/phoenix"
    report = write_detail_report(args.result_dir, phoenix_url=url, project_id=args.phoenix_project_id or os.getenv("BENCH_PHOENIX_PROJECT_ID", ""))
    print(json.dumps({"report": str(Path(args.result_dir).resolve() / "detail-report.html"), "complete_generated_matrix": report["complete_generated_matrix"], "responses": report["n_responses"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
