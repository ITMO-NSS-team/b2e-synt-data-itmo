"""Live, derived progress page for a long detail experiment; reads saved files only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import time
from .env import load_env


def _rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may be in the middle of its last append.
            continue
    return result


def write_progress(plan_path, results_root):
    plan_path, root = Path(plan_path).resolve(), Path(results_root).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    eval_id = plan_path.parent.name
    phase_root = root / ".phases" / eval_id
    status_path = phase_root / "experiment-status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    stopped = status.get("status") in {"cancelled", "interrupted"}
    rows, table = [], []
    phase_names = ["baseline", *["generated-generated_skills-" + group["slug"] for group in plan["groups"]]]
    groups = {"generated-generated_skills-" + group["slug"]: group for group in plan["groups"]}
    all_statuses = Counter()
    for phase in phase_names:
        folder = phase_root / phase / eval_id
        responses = _rows(folder / "responses.jsonl")
        scores = {row["run_id"]: row for row in _rows(folder / "scores.jsonl")}
        statuses = Counter(row["status"] for row in responses)
        all_statuses.update(statuses)
        quality = [scores[row["run_id"]]["metrics"].get("answer_accuracy") for row in responses
                   if row["status"] in {"completed", "normalization_pending"} and row["run_id"] in scores]
        quality = [value for value in quality if value is not None]
        accuracy = f"{100 * sum(quality) / len(quality):.1f}%" if quality else "—"
        group = groups.get(phase)
        label = group["generation_method"] + " / " + group["detail_level"] if group else "Baseline"
        table.append([label, str(len(responses)), str(len(quality)), accuracy, json.dumps(statuses, ensure_ascii=False)])
        rows += responses
    target = root / (eval_id + "-progress.html")
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    baseline_manifest = phase_root / "baseline" / eval_id / "run-manifest.json"
    baseline_modes, repetitions = 0, 1
    if baseline_manifest.exists():
        base = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        baseline_modes, repetitions = len(base["modes"]), int(base["repetitions"])
    expected = len(plan["selected_case_ids"]) * (len(plan["groups"]) + baseline_modes) * repetitions
    final_report = root / eval_id / "detail-report.html"
    report_link = f'<a href="{html.escape(eval_id)}/detail-report.html">Итоговый отчёт</a>' if final_report.exists() else "Итоговый отчёт появится после завершения всех фаз."
    if stopped:
        report_link = "Эксперимент остановлен. Данные неполные; итоговое сравнение не сформировано."
    refresh = '' if stopped or final_report.exists() else '<meta http-equiv="refresh" content="30">'
    state_message = ("Остановлен: " + html.escape(str(status.get("reason", "operator stop")))) if stopped else "Страница обновляется каждые 30 секунд."
    url = os.getenv("PUBLIC_URL", "https://localhost:8443").rstrip("/") + "/phoenix"
    body = "".join("<tr>" + "".join("<td>" + html.escape(cell) + "</td>" for cell in row) + "</tr>" for row in table)
    document = f"""<!doctype html><html lang="ru"><meta charset="utf-8">{refresh}<title>Benchmark progress</title>
<style>body{{font:16px system-ui;margin:32px;color:#182033}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #ddd}}th{{background:#f2f5fa}}</style>
<h1>Прогресс {html.escape(eval_id)}</h1><p>Завершено ячеек: {len(rows)} / {expected}. Кейсов в каждом generated-arm: {len(plan['selected_case_ids'])}. Обновление: {timestamp} UTC.</p>
<p>Промежуточные accuracy по разным фазам пока нельзя сравнивать: наборы завершённых кейсов могут различаться. Окончательный отчёт использует общие парные ячейки.</p>
<table><tr><th>Фаза</th><th>Запусков</th><th>Оценено</th><th>Текущая accuracy</th><th>Статусы</th></tr>{body}</table>
<p>{report_link} · <a href="{html.escape(url, quote=True)}">Phoenix: трейсы доступны уже сейчас</a></p><p>{state_message} Всего статусов: {html.escape(json.dumps(all_statuses))}.</p></html>"""
    temporary = target.with_suffix(".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(target)
    return final_report.exists() or stopped, target, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", default="benchmarking/results")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    load_env()
    while True:
        done, path, count = write_progress(args.plan, args.results)
        if done or not args.watch:
            print(json.dumps({"progress": str(path), "saved_cells": count, "final_report_ready": (Path(args.results) / path.stem.removesuffix("-progress") / "detail-report.html").exists(), "watch_finished": done}))
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
