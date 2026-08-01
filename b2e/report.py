"""Отчёт о правдоподобии корпуса: маргиналы и корреляции в человекочитаемом виде.

Гейт отвечает на вопрос «инварианты целы?». Этот отчёт отвечает на другой:
«похоже ли это на организацию?». Второе нельзя выразить булевой проверкой —
поэтому числа печатаются рядом с ожидаемым диапазоном, а решение остаётся за
читателем.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from heimdall.catalog import Catalog

from b2e.store import ProceduralSnapshot


def _pairs(x, y):
    ok = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(ok) < 3:
        return np.array([0.0]), np.array([0.0])
    xs, ys = zip(*ok)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def corr(x, y) -> float:
    xs, ys = _pairs(x, y)
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def realism_report(root: str | Path, catalog_path: str | Path) -> str:
    snap = ProceduralSnapshot(root, Catalog.load(catalog_path))
    t = snap.table("dm_core.employee_actual")
    out: list[str] = []

    def line(label: str, value: str, expect: str = "") -> None:
        out.append(f"  {label:<44} {value:>16}   {expect}")

    n = t.rows
    out.append(f"КОРПУС {snap.snapshot_id}: {n} сотрудников, "
               f"{snap.rows('dm_core.org_structure')} подразделений")
    out.append("")
    out.append("МАРГИНАЛЫ")
    status = Counter(t.column("employee_status"))
    line("активных", f"{100 * status['активный'] / n:.1f}%", "85–92%")
    line("фактическая численность (fact_flag)",
         f"{100 * np.mean(t.column('fact_flag')):.1f}%", "78–88%")
    line("руководителей", f"{100 * np.mean(t.column('position_flag_boss')):.1f}%",
         "5–15%")
    line("на испытательном сроке",
         f"{100 * np.mean(t.column('is_probation')):.2f}%", "1–5%")
    age = np.asarray(t.column("age"), dtype=float)
    line("возраст: медиана / p90", f"{np.median(age):.0f} / {np.percentile(age, 90):.0f}",
         "33–38 / 48–54")
    ten = np.asarray(t.column("experience_sber_year"), dtype=float)
    line("стаж в компании: медиана", f"{np.median(ten):.1f} лет", "2–5")
    grades = Counter(t.column("grade_level"))
    top = max(grades.values()) / n
    line("самая крупная корзина грейда", f"{100 * top:.1f}%", "< 25%")
    line("женщин", f"{100 * np.mean([g == 'Ж' for g in t.column('gender')]):.1f}%",
         "55–68%")

    out.append("")
    out.append("СВЯЗИ (корреляция Пирсона)")
    grade = t.column("grade_level")
    cc = snap.table("dm_special.employee_competence_actual")
    comp = cc.column("soft_skills_competency_score")
    marks = t.column("estimation_q1")
    letter = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    mark_num = [letter.get((m or " ").split()[0]) if m else None for m in marks]
    line("возраст ↔ грейд", f"{corr(age, grade):.3f}", "0.30–0.50")
    line("компетенции ↔ грейд", f"{corr(comp, grade):.3f}", "0.40–0.60")
    line("оценка ↔ грейд", f"{corr(mark_num, grade):.3f}", "0.20–0.45")
    line("оценка ↔ компетенции", f"{corr(mark_num, comp):.3f}", "0.30–0.55")

    out.append("")
    out.append("СЛОВАРИ")
    names = t.column("employee_full_name")
    surnames = [x.split()[0] for x in names if x]
    line("уникальных фамилий", f"{len(set(surnames))}", "чем больше, тем лучше")
    line("полных тёзок", f"{len(names) - len(set(names))}", "единицы, не сотни")
    line("в верхнем регистре (каверза ILIKE)",
         f"{100 * np.mean([x == x.upper() for x in names if x]):.1f}%", "0% или задано")

    ed = snap.table("dm_core.employee_actual")
    uni = [v for row in ed.column("educational_institution_name") for v in (row or [])]
    line("вузов различных", f"{len(set(uni))}", "> 50")
    line("курсов различных",
         f"{len(set(snap.table('dm_core.employee_education').column('item_name')))}",
         "> 50")

    out.append("")
    out.append("ВИТРИНЫ")
    for key in snap.keys():
        rows = snap.rows(key)
        flag = "  ← пусто" if rows == 0 else ""
        out.append(f"  {rows:>9}  {key}{flag}")
    return "\n".join(out)
