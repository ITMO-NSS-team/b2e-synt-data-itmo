"""Цепочка разрешения колонки: одно имя — одно значение на всех витринах.

Главная идея проекта
--------------------
В прежнем корпусе каждая витрина заполнялась независимо, а «известными» были
несколько перечисленных вручную колонок. Отсюда следствие, которое ломает любой
сценарий агента: ``employee_competence_actual.employee_full_name`` совпадал с
``employee_actual`` на **0 строках из 3000**. Один и тот же ``person_id`` носил
два разных имени, и заметить это изнутри API было нельзя.

Здесь витрина — не генератор, а **проекция**. Любая колонка проходит цепочку:

1. явное переопределение витрины      (``position_actual.vacancy``)
2. атрибут человека по имени/алиасу    (``employee_full_name``, ``grade_level``)
3. атрибут оргединицы по уровню        (``oshs_level_7_unit_name``)
4. объявленная группа массивов         (``educ.*``, ``educational_*``)
5. процедурный заполнитель по имени    (``*_email``, ``*_dt``)
6. процедурный заполнитель по типу

Добавление витрины не требует ни строчки маппинга: она наследует всё, что
называется так же. Один новый алиас чинит целый класс колонок сразу на всех 37
витринах — а не на одной, где о нём вспомнили.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import dicts, fillers
from .blocks import Block
from .population import AS_OF_DAYS, Population, _days_to_iso, position_name
from .rng import bernoulli, cells, integers, key64, normal, pick, unit

#: Колонки уровней оргструктуры: ``oshs_level_7_unit_name_main`` и родня.
_ORG_LEVEL = re.compile(
    r"^(oshs|fos|agile|fg|b2c|tb)_level_(\d{1,2})_unit_(name|id)(_main)?$")
#: Иерархические перечисления руководителей: ``agile_director_hierarchy_person_id``.
_ORG_CHIEF = re.compile(
    r"^(oshs|fos|agile|fg)_(curator|director|hrbp|owner)"
    r"(_hierarchy)?_(person_id|employee_id|full_name)(_main)?$")

#: Плоские имена, за которыми стоит группа массивов образования.
_EDU_FLAT = {
    "educational_institution_name": ("education", "institution"),
    "educational_institution": ("education", "institution"),
    "educational_speciality": ("education", "speciality"),
    "educational_faculty": ("education", "speciality"),
    "education_type_name": ("education", "level"),
    "education_name": ("education", "level"),
    "institute_name": ("education", "institution"),
    "specialty": ("education", "speciality"),
    "education_end_year": ("education", "end_year"),
    "education_end_month": ("education", "end_month"),
    "other_education_course": ("learning", "item_name"),
    "other_education_end_year": ("learning", "assigned_days"),
    "other_education_place": ("education", "institution_city"),
}

#: Блоки каталога с точкой в имени: ``educ.item_name`` → блок learning, поле item_name.
_DOTTED = {
    "educ": ("learning", {
        "assignment_id": "assignment_id", "assignment_status": "assignment_status",
        "assignment_type": "assignment_type", "item_name": "item_name",
        "item_type": "item_type", "item_format_name": "item_format_name",
        "item_duration": "item_duration", "item_assigned_dt": "assigned_days",
        "item_deadline_dt": "deadline_days", "item_completed_dt": "completed_days",
        "playlist_name": "item_name", "playlist_assigned_dt": "assigned_days",
        "playlist_deadline_dt": "deadline_days",
        "playlist_completed_dt": "completed_days",
    }),
    "estimation": ("estimation", {
        "year": "year", "quarter": "quarter", "performance": "performance",
        "standards": "standards",
    }),
    "goals": ("goals", {
        "title": "title", "type": "type", "quarter": "quarter",
        "progress": "progress", "tags": "tags", "risk_state": "risk",
    }),
    "goal": ("goals", {"title": "title", "type": "type", "progress": "progress"}),
    "absence": ("absence", {
        "absence_name": "name", "absence_group_name": "group_name",
        "absence_code": "code", "absence_group": "group",
        "absence_start": "start_days", "absence_end": "end_days",
        "flag_blocked": "blocked",
    }),
    "successors": ("successors", {
        "status": "status", "appoint_date": "appoint_days",
    }),
    "predecessor": ("successors", {"status": "status"}),
    "achievements": ("achievements", {
        "name": "name", "description": "description", "period": "period",
    }),
    "career": ("career", {
        "company": "company", "position": "position",
        "start_date": "start_days", "end_date": "end_days",
    }),
}

#: Поля блоков, которые нужно отдать как дату, а не как число дней.
_DAY_FIELDS = {"assigned_days", "deadline_days", "completed_days", "start_days",
               "end_days", "appoint_days"}


@dataclass
class Frame:
    """Строковое пространство витрины.

    ``person`` — номер человека для каждой строки витрины (или -1, если строка
    не о человеке). Всё, что относится к людям, разрешается через него, поэтому
    личность на любой витрине берётся из одного источника.
    """

    key: str
    n: int
    person: np.ndarray
    extra: dict = field(default_factory=dict)
    unit: np.ndarray | None = None


class Resolver:
    """Разрешение колонки каталога в значения для конкретной витрины."""

    def __init__(self, pop: Population, blocks: dict[str, Block], seed: int) -> None:
        self.p = pop
        self.blocks = blocks
        self.seed = seed
        self.tree = pop.tree
        self._ancestors = pop.tree.ancestors_matrix()
        self._position_name = None
        self._ref_cache: dict[str, np.ndarray] = {}
        self._aliases = _build_aliases()

    # ------------------------------------------------------------------ ядро
    def column(self, model_key: str, member, frame: Frame,
               strict: bool = False) -> list | None:
        """Значение колонки. ``strict`` — не подставлять процедурный хвост.

        Строгий режим нужен сборщику: колонки, которые цепочка не знает, на диск
        не пишутся вовсе и вычисляются на чтении. Иначе 300 000 × 3 800 хвостовых
        колонок пришлось бы материализовать — 1,14 млрд ячеек.
        """
        name = member.name
        if name in frame.extra:
            value = frame.extra[name]
            return value.tolist() if isinstance(value, np.ndarray) else list(value)

        rows = frame.person
        known = rows >= 0
        if known.any():
            got = self._person_column(name, member, frame)
            if got is not None:
                return got
        got = self._org_column(name, member, frame)
        if got is not None:
            return got
        if strict:
            return None
        return fillers.column(name, member.ch, model_key, self.seed,
                              np.arange(frame.n), self.p.book)

    # -------------------------------------------------------------- человек
    def _person_column(self, name: str, member, frame: Frame):
        rows = np.where(frame.person >= 0, frame.person, 0)
        mask = frame.person >= 0

        fn = self._aliases.get(name)
        if fn is not None:
            values = fn(self, rows)
            return _mask_out(values, mask, member)

        block_ref = _EDU_FLAT.get(name)
        if block_ref is None and "." in name:
            head, tail = name.split(".", 1)
            spec = _DOTTED.get(head)
            if spec and tail in spec[1]:
                block_ref = (spec[0], spec[1][tail])
        if block_ref is not None:
            return self._block_column(block_ref, rows, mask, member)
        return None

    def _block_column(self, ref: tuple[str, str], rows: np.ndarray,
                      mask: np.ndarray, member):
        block = self.blocks[ref[0]]
        field_name = ref[1]
        if field_name == "row":                       # ссылка на другого человека
            lists = block.lists("row", rows)
            return [[self.p["full_name"][r] for r in ids] if ok else []
                    for ids, ok in zip(lists, mask)]
        lists = block.lists(field_name, rows)
        if field_name in _DAY_FIELDS:
            lists = [[None if d < 0 else _iso(d) for d in item] for item in lists]
        if not getattr(member.ch, "is_array", False):
            # Витрина-развёртка просит скаляр: берётся первая запись.
            return [item[0] if item else None for item in lists]
        return [item if ok else [] for item, ok in zip(lists, mask)]

    # ------------------------------------------------------ оргструктура
    def _org_column(self, name: str, member, frame: Frame):
        units = frame.unit
        if units is None and (frame.person >= 0).any():
            units = self.p["unit_id"][np.where(frame.person >= 0, frame.person, 0)]
        if units is None:
            return None

        m = _ORG_LEVEL.match(name)
        if m:
            level = int(m.group(2))
            if level > self._ancestors.shape[1]:
                return [None] * frame.n
            node = self._ancestors[units, level - 1]
            if m.group(3) == "id":
                # np.where по всему массиву дешевле, чем ветвление в цикле:
                # колонок уровней пятнадцать, и каждая — 294 000 строк.
                ids = np.where(node >= 0, 1_000_000 + node, -1)
                return [None if v < 0 else str(v) for v in ids]
            names = np.where(node >= 0, self.tree.name[np.maximum(node, 0)], None)
            return names.tolist()

        m = _ORG_CHIEF.match(name)
        if m:
            what = m.group(4)
            # Значение считается на УНИКАЛЬНОЕ подразделение, а не на строку.
            # Подразделений 32 500 против 294 000 человек, а цепочка руководителей
            # — обход дерева: наивный вариант давал миллионы обходов на колонку
            # и растягивал сборку на часы.
            uniq, inverse = np.unique(units, return_inverse=True)
            if m.group(3):                       # иерархия: цепочка руководителей
                per_unit = [self._chain(int(u), what) for u in uniq]
                return [list(per_unit[i]) for i in inverse]
            heads = self.tree.head_person[uniq]
            per_unit = [self._person_ref(int(h), what) for h in heads]
            return [per_unit[i] for i in inverse]
        return None

    def _refs(self, what: str) -> np.ndarray:
        """Массив ссылок на людей нужного вида — считается один раз на корпус."""
        cached = self._ref_cache.get(what)
        if cached is None:
            if what == "person_id":
                cached = person_uuid(self.p, np.arange(self.p.n))
            elif what == "employee_id":
                cached = np.array([str(int(v)) for v in self.p["employee_id"]],
                                  dtype=object)
            else:
                cached = self.p["full_name"]
            self._ref_cache[what] = cached
        return cached

    def _chain(self, unit_id: int, what: str) -> list:
        refs = self._refs(what)
        out = []
        node = int(unit_id)
        while node >= 0 and len(out) < 8:
            head = int(self.tree.head_person[node])
            if head >= 0:
                out.append(refs[head])
            node = int(self.tree.parent[node])
        return out

    def _person_ref(self, row: int, what: str):
        if row < 0:
            return None
        return self._refs(what)[row]

    # ------------------------------------------------------------- сервис
    def position_names(self) -> np.ndarray:
        if self._position_name is None:
            self._position_name = position_name(self.p)
        return self._position_name


def _iso(day: int) -> str:
    return _days_to_iso(np.array([day]))[0]


def _mask_out(values, mask: np.ndarray, member):
    out = values.tolist() if isinstance(values, np.ndarray) else list(values)
    if mask.all():
        return out
    empty = [] if getattr(member.ch, "is_array", False) else None
    return [v if ok else empty for v, ok in zip(out, mask)]


def person_uuid(p: Population, rows: np.ndarray) -> np.ndarray:
    """Устойчивый ``person_id``: один человек — один UUID во всех витринах."""
    a = cells(key64(p.seed, "person.uuid.a"), rows)
    b = cells(key64(p.seed, "person.uuid.b"), rows)
    return np.array([f"{int(x) >> 32:08x}-{int(x) >> 16 & 0xffff:04x}-"
                     f"4{int(x) & 0xfff:03x}-a{int(y) >> 48 & 0xfff:03x}-"
                     f"{int(y) & 0xffffffffffff:012x}" for x, y in zip(a, b)],
                    dtype=object)


# ----------------------------------------------------------------- алиасы

def _build_aliases() -> dict:
    """Имя колонки каталога → как взять значение у человека.

    Таблица намеренно плоская и данными, а не кодом: её читают и дополняют чаще,
    чем любой другой файл проекта.
    """
    A: dict = {}

    def reg(*names):
        def deco(fn):
            for nm in names:
                A[nm] = fn
            return fn
        return deco

    # --- идентификация
    reg("person_id")(lambda r, x: person_uuid(r.p, x))
    reg("employee_id", "tn", "tab_num", "personnel_number")(
        lambda r, x: np.array([str(int(v)) for v in r.p["employee_id"][x]], dtype=object))
    reg("emp_key")(lambda r, x: np.array(
        [[dicts.COMPANIES[c], str(int(e))] for c, e in
         zip(r.p["company_idx"][x], r.p["employee_id"][x])], dtype=object))
    reg("emp_key_str")(lambda r, x: np.array(
        [f"{dicts.COMPANIES[c]}:{int(e)}" for c, e in
         zip(r.p["company_idx"][x], r.p["employee_id"][x])], dtype=object))
    reg("company")(lambda r, x: np.array(dicts.COMPANIES, dtype=object)[
        r.p["company_idx"][x]])
    reg("company_name")(lambda r, x: np.array(dicts.COMPANY_NAMES, dtype=object)[
        r.p["company_idx"][x]])
    reg("is_dzo")(lambda r, x: (r.p["company_idx"][x] > 0).astype(np.int64))

    # --- имя
    reg("employee_full_name", "full_name", "fio", "employee_fio")(
        lambda r, x: r.p["full_name"][x])
    reg("employee_first_name", "first_name")(lambda r, x: r.p["first_name"][x])
    reg("employee_last_name", "last_name")(lambda r, x: r.p["last_name"][x])
    reg("employee_mid_name", "mid_name", "middle_name")(
        lambda r, x: r.p["middle_name"][x])
    reg("employee_short_name", "short_name")(lambda r, x: np.array(
        [f"{l} {f[0]}.{m[0]}." for l, f, m in zip(r.p["last_name"][x],
                                                  r.p["first_name"][x],
                                                  r.p["middle_name"][x])],
        dtype=object))
    reg("gender", "sex")(lambda r, x: np.where(r.p["female"][x], "Ж", "М"))

    # --- возраст и даты
    reg("age")(lambda r, x: r.p["age"][x])
    reg("age_is_adult")(lambda r, x: (r.p["age"][x] >= 18).astype(np.int64))
    reg("birthday", "birth_date", "birthday_date")(
        lambda r, x: _days_to_iso(r.p["birth_days"][x]))
    reg("birth_year")(lambda r, x: r.p["birth_year"][x])
    reg("generation")(lambda r, x: np.array(
        ["Беби-бумеры", "X", "Y", "Z"], dtype=object)[
            np.digitize(r.p["birth_year"][x], [1965, 1981, 1997])])
    reg("last_hire_date", "hire_date", "employment_date")(
        lambda r, x: _days_to_iso(r.p["hire_days"][x]))
    reg("last_fired_date", "dismissal_date")(lambda r, x: [
        None if d < 0 else _iso(d) for d in r.p["fired_days"][x]])
    reg("position_start_date", "position_date")(
        lambda r, x: _days_to_iso(r.p["position_start_days"][x]))
    reg("report_date")(lambda r, x: np.array(
        [_iso(AS_OF_DAYS)] * len(x), dtype=object))

    # --- стаж
    reg("experience_sber_year", "experience_nonstop_sber_year")(
        lambda r, x: r.p["tenure_years"][x].astype(np.int64))
    reg("experience_sber_month", "experience_nonstop_sber_month")(
        lambda r, x: (r.p["tenure_years"][x] * 12).astype(np.int64))
    reg("experience_in_position_year")(
        lambda r, x: r.p["role_years"][x].astype(np.int64))
    reg("experience_in_position_month")(
        lambda r, x: (r.p["role_years"][x] * 12).astype(np.int64))
    reg("experience_total_year", "total_experience_years")(
        lambda r, x: r.p["total_exp_years"][x].astype(np.int64))

    # --- статус
    reg("employee_status", "status")(lambda r, x: np.where(
        r.p["fired"][x], "уволен", "активный"))
    reg("fact_flag")(lambda r, x: r.p["fact_flag"][x])
    reg("form_work")(lambda r, x: np.array([f[0] for f in dicts.FORM_WORK],
                                           dtype=object)[r.p["form_idx"][x]])
    reg("form_work_code")(lambda r, x: r.p["form_idx"][x] + 1)
    reg("work_mode")(lambda r, x: np.array([w[0] for w in dicts.WORK_MODE],
                                           dtype=object)[r.p["work_mode_idx"][x]])
    reg("is_probation", "probation_flag")(lambda r, x: r.p["probation"][x])
    reg("intern_flag", "is_intern")(lambda r, x: r.p["intern"][x])

    # --- должность и грейд
    reg("grade_level", "grade_level_position", "pos_grade_num")(
        lambda r, x: r.p["grade_level"][x])
    reg("grade", "grade_position")(lambda r, x: np.array(
        [f"G{g}" for g in r.p["grade_level"][x]], dtype=object))
    reg("position_id")(lambda r, x: np.array(
        [str(int(v)) for v in r.p["position_id"][x]], dtype=object))
    reg("position_name", "position_name_main", "job_title")(
        lambda r, x: r.position_names()[x])
    reg("position_flag_boss", "is_manager", "boss_flag")(
        lambda r, x: r.p["is_head"][x])
    reg("type_position_id")(lambda r, x: r.p["family_idx"][x] + 1)
    reg("job_family", "position_family")(lambda r, x: np.array(
        list(r.p["family_names"]), dtype=object)[r.p["family_idx"][x]])

    # --- организация
    reg("block_name", "fblock_name", "blok_name")(lambda r, x: np.array(
        [b[0] for b in dicts.BLOCKS], dtype=object)[r.p["block_idx"][x]])
    reg("tb_name", "territorial_bank", "tb")(lambda r, x: np.array(
        [t[0] for t in dicts.TERRITORIAL_BANKS] + ["—"], dtype=object)[
            np.where(r.p["tb_idx"][x] >= 0, r.p["tb_idx"][x], len(dicts.TERRITORIAL_BANKS))])
    reg("unit_name", "unit_b2c_name", "oshs_unit_name", "division_name")(
        lambda r, x: r.tree.name[r.p["unit_id"][x]])
    reg("unit_id", "oshs_unit_id", "org_unit_id")(lambda r, x: np.array(
        [str(1_000_000 + int(u)) for u in r.p["unit_id"][x]], dtype=object))
    reg("city_name", "city", "work_city", "gosb_name")(lambda r, x: np.array(
        [c[0] for c in dicts.CITIES], dtype=object)[r.p["city_idx"][x]])
    reg("expertise_area", "domain", "primary_domain")(lambda r, x: np.array(
        list(r.p["family_names"]), dtype=object)[r.p["family_idx"][x]])

    # --- личное
    reg("family_status", "marital_status")(lambda r, x: np.array(
        [f[0] for f in dicts.FAMILY_STATUS], dtype=object)[r.p["family_status_idx"][x]])
    reg("children_qty", "child_count", "children_count")(
        lambda r, x: r.p["children_qty"][x])
    reg("mobility_status")(lambda r, x: np.array(
        [m[0] for m in dicts.MOBILITY_STATUS], dtype=object)[r.p["mobility_idx"][x]])
    reg("career_status")(lambda r, x: np.array(dicts.CAREER_STATUS, dtype=object)[
        r.p["career_status_idx"][x]])
    reg("vacation_days_balance")(lambda r, x: (
        14 + (r.p["tenure_years"][x] * 3).astype(np.int64) % 20))

    # --- оценка и потенциал
    reg("engagement", "engagement_index", "engagement_score")(
        lambda r, x: r.p["engagement"][x])
    reg("star_index")(lambda r, x: [None if v < 0 else int(v)
                                    for v in r.p["star_index"][x]])
    reg("score_total")(lambda r, x: [None if v < 0 else int(v)
                                     for v in r.p["star_index"][x]])
    reg("has_embedding")(lambda r, x: r.p["has_embedding"][x])
    reg("is_key_employee", "key_employee_flag")(lambda r, x: _key_employee(r, x))
    reg("talent_pool_flag", "is_talent_pool", "kadrovyi_rezerv")(
        lambda r, x: r.p["talent_pool"][x])
    reg("attrition_risk", "churn_risk", "risk_level")(lambda r, x: np.array(
        ["Низкий", "Средний", "Высокий"], dtype=object)[r.p["attrition_risk_idx"][x]])

    # --- роллапы блоков
    reg("num_goals", "goals_qty")(lambda r, x: r.blocks["goals"].counts[x])
    reg("risked_goals")(lambda r, x: _risked_goals(r, x))
    reg("mean_value_completion", "goals_completion")(
        lambda r, x: _mean_progress(r, x))
    reg("goals_last_quarter")(lambda r, x: _last_quarter(r, x))
    reg("successors_qty")(lambda r, x: r.blocks["successors"].counts[x])
    reg("num_absence", "absence_qty")(lambda r, x: r.blocks["absence"].counts[x])
    reg("education_qty", "num_education")(lambda r, x: r.blocks["education"].counts[x])
    for q in (1, 2, 3, 4):
        A[f"estimation_q{q}"] = (lambda q: lambda r, x: _mark_of(r, x, q))(q)
    reg("estimation_year")(lambda r, x: _mark_year(r, x))
    return A


def _key_employee(r: "Resolver", x: np.ndarray) -> np.ndarray:
    """Метка ключевого сотрудника — ПРОШЛОГОДНИЙ срез методики.

    Она намеренно не равна расчёту по текущим данным: агент, взявший флаг вместо
    расчёта, разойдётся с эталоном, и рубрика это поймает. Без расхождения
    корзина не различает знание методики и угадывание.
    """
    perf = r.p.truth["perf_latent"][x] + 0.35 * normal(
        key64(r.seed, "key.lastyear"), x)
    comp = r.p.truth["competency_avg"][x]
    return ((perf > 0.75) & (comp >= 3.6)).astype(np.int64)


#: Свёртки блоков считаются по всему корпусу сразу и кэшируются: на 294 000
#: человек поэлементный цикл на каждую колонку стоит дороже самих данных.

def _rollup(r: "Resolver", name: str, fn):
    cached = r._ref_cache.get(name)
    if cached is None:
        cached = r._ref_cache[name] = fn()
    return cached


def _risked_goals(r: "Resolver", x: np.ndarray) -> np.ndarray:
    b = r.blocks["goals"]

    def compute():
        rows = b.flat_rows()
        return np.bincount(rows[b.fields["progress"] < 0.5],
                           minlength=len(b.counts))
    return _rollup(r, "risked_goals", compute)[x]


def _mean_progress(r: "Resolver", x: np.ndarray):
    b = r.blocks["goals"]

    def compute():
        rows = b.flat_rows()
        total = np.bincount(rows, weights=b.fields["progress"],
                            minlength=len(b.counts))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(b.counts > 0, total / np.maximum(b.counts, 1), np.nan)
        return np.round(mean, 4)
    values = _rollup(r, "mean_progress", compute)[x]
    return [None if np.isnan(v) else float(v) for v in values]


def _last_quarter(r: "Resolver", x: np.ndarray):
    b = r.blocks["goals"]

    def compute():
        out = np.zeros(len(b.counts), dtype=np.int64)
        np.maximum.at(out, b.flat_rows(), b.fields["quarter"])
        return out
    values = _rollup(r, "last_quarter", compute)[x]
    return [None if c == 0 else int(v) for v, c in zip(values, b.counts[x])]


def _mark_of(r: "Resolver", x: np.ndarray, quarter: int):
    """Оценка за квартал — свёртка блока estimation, а не независимая величина."""
    b = r.blocks["estimation"]

    def compute():
        rows, qq = b.flat_rows(), b.fields["quarter"]
        out = np.full(len(b.counts), None, dtype=object)
        hit = qq == quarter
        # Записи идут по возрастанию периода, поэтому присваивание оставляет
        # последнюю — то есть самую свежую оценку за этот квартал.
        out[rows[hit]] = b.fields["performance"][hit]
        return out
    return _rollup(r, f"mark_q{quarter}", compute)[x].tolist()


def _mark_year(r: "Resolver", x: np.ndarray):
    b = r.blocks["estimation"]

    def compute():
        out = np.full(len(b.counts), None, dtype=object)
        last = b.offsets[b.counts > 0] + b.counts[b.counts > 0] - 1
        out[b.counts > 0] = b.fields["performance"][last]
        return out
    return _rollup(r, "mark_year", compute)[x].tolist()
