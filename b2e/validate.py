"""Гейт согласованности: инварианты корпуса проверяются, а не декларируются.

Каждая проверка здесь соответствует измеренному дефекту прежнего корпуса. Это
не абстрактная гигиена: под каждым номером лежит число, полученное на реальных
данных предшественника, и проверка существует, чтобы оно не вернулось.

Гейт возвращает ненулевой код при любом ОТКАЗЕ. Предупреждения (WARN) не валят
сборку: это диапазоны правдоподобия, а не инварианты.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from heimdall.catalog import Catalog

from b2e.store import ProceduralSnapshot

#: Витрины, чьи строки — люди: на них проверяется единство личности.
PERSON_MARTS = [
    "dm_core.employee_actual", "dm_core.employee", "dm_core.employee_hist",
    "dm_special.employee_competence_actual", "dm_special.churn_model_metrics",
    "dm_core.employee_oshs", "dm_special.top_600_stats",
    "dm_special.staff_employee_digital_profile_top_600",
]

#: Группа образования: длины массивов обязаны совпадать внутри строки.
EDU_GROUP = ["educational_institution_name", "educational_speciality",
             "education_type_name"]


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append(("OK" if ok else "FAIL", name, detail))
        return ok

    def warn(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append(("OK" if ok else "WARN", name, detail))

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == "FAIL")

    def render(self) -> str:
        width = max(len(n) for _, n, _ in self.rows) + 2
        lines = [f"{s:<5} {n:<{width}} {d}" for s, n, d in self.rows]
        warns = sum(1 for s, _, _ in self.rows if s == "WARN")
        lines.append("")
        lines.append(f"проверок: {len(self.rows)}, отказов: {self.failed}, "
                     f"предупреждений: {warns}")
        return "\n".join(lines)


def run(root: str | Path, catalog_path: str | Path = "catalog/snapshot.json") -> Report:
    snap = ProceduralSnapshot(root, Catalog.load(catalog_path))
    rep = Report()
    base = snap.table("dm_core.employee_actual")
    ids = base.column("person_id")
    names = base.column("employee_full_name")
    by_id = dict(zip(ids, names))

    # --- W1: одна личность на всех витринах ------------------------------
    for key in PERSON_MARTS[1:]:
        if not snap.has(key):
            continue
        t = snap.table(key)
        if "employee_full_name" not in t.available():
            continue
        pid = t.column("person_id")
        nm = t.column("employee_full_name")
        pairs = [(p, n) for p, n in zip(pid, nm) if p in by_id]
        # Регистр сравнивается без учёта: каверза ``upper_cyrillic`` меняет
        # написание, но обязана делать это одинаково на всех витринах — что и
        # проверяется отдельно ниже.
        agree = sum(1 for p, n in pairs
                    if (by_id[p] or "").casefold() == (n or "").casefold())
        rep.check(f"W1 личность едина: {key}", pairs and agree == len(pairs),
                  f"{agree}/{len(pairs)}")

    # --- W2: части имени согласованы с целым -----------------------------
    for key in ("dm_core.employee_actual", "dm_core.employee_hist"):
        t = snap.table(key)
        full = t.column("employee_full_name")
        last = t.column("employee_last_name")
        ok = sum(1 for f, l in zip(full, last)
                 if f and l and f.split()[0].casefold() == l.casefold())
        rep.check(f"W2 фамилия = первый токен ФИО: {key}", ok == len(full),
                  f"{ok}/{len(full)}")

    # --- W2b: дата рождения согласована с возрастом ----------------------
    age = base.column("age")
    bday = base.column("birthday")
    bad = sum(1 for a, b in zip(age, bday)
              if b and abs((2026 - int(b[:4])) - int(a)) > 1)
    rep.check("W2 возраст = дата рождения", bad == 0, f"расхождений {bad}")

    # --- W3: группы массивов выровнены -----------------------------------
    t = snap.table("dm_core.employee_actual")
    cols = [t.column(c) for c in EDU_GROUP if c in t.available()]
    if len(cols) >= 2:
        aligned = sum(1 for row in zip(*cols)
                      if len({len(v or []) for v in row}) == 1)
        rep.check("W3 образование выровнено позиционно", aligned == len(cols[0]),
                  f"{aligned}/{len(cols[0])}")

    # --- W5: компетенции различимы ---------------------------------------
    cc = snap.table("dm_special.employee_competence_actual")
    comp_names = ["personality_traits_group_score", "cognitive_features_group_score",
                  "soft_skills_competency_score", "management_competency_score",
                  "wide_context", "influence_scale", "reflection"]
    have = [cc.column(c) for c in comp_names if c in cc.available()]
    if len(have) >= 3:
        spread = [max(r) - min(r) for r in zip(*have)]
        rep.check("W5 компетенции различимы внутри человека",
                  float(np.mean(spread)) > 0.3,
                  f"средний разброс {np.mean(spread):.2f}")

    # --- W6: правдоподобные маргиналы ------------------------------------
    boss = np.array(t.column("position_flag_boss"), dtype=float)
    rep.warn("W6 доля руководителей 5–15%", 0.05 <= boss.mean() <= 0.15,
             f"{100 * boss.mean():.1f}%")
    grades = Counter(t.column("grade_level"))
    top_share = max(grades.values()) / sum(grades.values())
    rep.warn("W6 нет свалки на потолке грейда", top_share < 0.25,
             f"максимальная корзина {100 * top_share:.1f}%")
    prob = np.array(t.column("is_probation"), dtype=float)
    rep.check("W6 испытательный срок не мёртвая колонка", 0 < prob.mean() < 0.15,
              f"{100 * prob.mean():.2f}%")

    # --- W7: ссылочная целостность ---------------------------------------
    org = snap.table("dm_core.org_structure")
    if "oshs_hrbp_employee_id" in org.available():
        emp_ids = set(str(v) for v in t.column("employee_id"))
        heads = [v for v in org.column("oshs_hrbp_employee_id") if v]
        hit = sum(1 for h in heads if str(h) in emp_ids)
        rep.check("W7 руководители подразделений — реальные сотрудники",
                  heads and hit == len(heads), f"{hit}/{len(heads)}")

    # --- W10: пустых витрин нет, кроме объявленных ------------------------
    from b2e.gen.marts import EMPTY_BY_DESIGN
    empty = [k for k in snap.keys() if snap.rows(k) == 0]
    rep.check("W10 пустых витрин нет сверх объявленных",
              set(empty) <= EMPTY_BY_DESIGN, f"пусто: {sorted(empty)}")

    # --- W11: история сворачивается в текущее состояние -------------------
    hist = snap.table("dm_core.employee_hist")
    if "_changes" in hist.available():
        changes = hist.column("_changes")
        grade_now = hist.column("grade_level")
        mismatch = 0
        for events, current in zip(changes, grade_now):
            for _, column, value in events or []:
                if column == "grade_level" and value != current:
                    mismatch += 1
        rep.check("W11 события истории сворачиваются в текущее состояние",
                  mismatch == 0, f"расхождений {mismatch}")
        total = sum(len(e or []) for e in changes)
        rep.warn("W11 история непуста", total > 0, f"событий {total}")

    # --- каверзы: объявленные присутствуют и согласованы ------------------
    enabled = set(snap.manifest.get("traps", []))
    if "upper_cyrillic" in enabled:
        upper = [x for x in names if x and x == x.upper()]
        rep.check("каверза upper_cyrillic присутствует", len(upper) > 0,
                  f"{100 * len(upper) / len(names):.1f}% записей")
        other = snap.table("dm_special.employee_competence_actual")
        if "employee_full_name" in other.available():
            pid2 = other.column("person_id")
            nm2 = dict(zip(pid2, other.column("employee_full_name")))
            same = sum(1 for p, n in zip(ids, names)
                       if p not in nm2 or nm2[p] == n)
            rep.check("каверза одинакова на всех витринах", same == len(ids),
                      f"{same}/{len(ids)}")

    # --- W12: истина отделена от витрин ----------------------------------
    truth = Path(root) / "truth" / "people.json"
    rep.check("W12 истина сохранена отдельно", truth.exists(), str(truth))
    leaked = [c for c in t.available() if c.endswith("_hidden") or c == "ability"]
    rep.check("W12 скрытые факторы не утекли в витрину", not leaked, str(leaked))

    # --- каталог: имена колонок принадлежат каталогу ----------------------
    catalog = snap.catalog
    stray = []
    for key in snap.keys():
        model = catalog.models.get(key)
        if model is None:
            continue
        declared = set(model.columns) | {"_changes"}
        stray += [f"{key}.{c}" for c in snap.table(key)._index if c not in declared]
    rep.check("каталог: посторонних колонок нет", not stray, str(stray[:5]))
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Гейт согласованности корпуса")
    ap.add_argument("--data", default="data")
    ap.add_argument("--catalog", default="catalog/snapshot.json")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    rep = run(args.data, args.catalog)
    print(rep.render())
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"status": s, "check": n, "detail": d} for s, n, d in rep.rows],
            ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
