"""Ядро популяции: один человек — одна строка, все витрины его проекции.

Три вещи, которые здесь делаются иначе, чем в прежнем генераторе
----------------------------------------------------------------
**1. Квантильное отображение вместо обрезки.** Грейд прежде считался как
``clamp(8 + 0.45·стаж + 1.1·способность)`` и давал 12,8% людей на потолке G18 и
корреляцию с возрастом 0,88. Здесь скрытая «старшинство» отображается на
*заданное* распределение грейдов по квантилям: маргинальное распределение
получается ровно таким, каким задумано, а связь со способностью сохраняется.

**2. Оценки — порядковая модель, а не независимый выбор.** Прежде метки A–E
брались из фиксированных весов и не коррелировали ни с чем (corr с грейдом
0,035). Здесь пороги режут скрытую результативность, поэтому и целевое
распределение (принудительное ранжирование), и связь со способностью
выполняются одновременно.

**3. Компетенции различимы внутри человека.** Прежде девять колонок были
побайтово равны. Здесь у каждой компетенции своя нагрузка на способность и своя
идиосинкразия — «в чём человек слабее» имеет ответ.

Латентные факторы остаются скрытыми: они уходят в ``truth/``, но ни в одну
витрину не попадают.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from . import dicts, org
from .names import NameBook
from .rng import (bernoulli, cells, exponential, integers, key64, normal, pick,
                  unit, weighted)

AS_OF = dt.date(2026, 7, 1)
EPOCH = dt.date(1970, 1, 1)
AS_OF_DAYS = (AS_OF - EPOCH).days

#: Целевая пирамида грейдов: доля численности на каждом грейде 6..20.
GRADE_LEVELS = np.arange(6, 21)
GRADE_SHARE = np.array([0.02, 0.05, 0.11, 0.15, 0.16, 0.14, 0.11, 0.08,
                        0.055, 0.04, 0.028, 0.018, 0.011, 0.006, 0.002])

#: Возрастные корзины и их доли — профиль крупного российского банка.
AGE_BUCKETS = [(18, 22, 0.03), (22, 26, 0.13), (26, 30, 0.18), (30, 35, 0.21),
               (35, 40, 0.17), (40, 45, 0.11), (45, 50, 0.08), (50, 55, 0.05),
               (55, 60, 0.03), (60, 66, 0.01)]

#: Годовая текучесть — доля уволенных в снимке за последние 12 месяцев.
ANNUAL_ATTRITION = 0.12

#: Доля людей без оценки силы профиля: NULL значит «не скорен», не «слаб».
UNSCORED_RATE = 0.17


def _cum(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    return np.cumsum(w / w.sum())


def _quantile_map(latent: np.ndarray, values: np.ndarray,
                  share: np.ndarray) -> np.ndarray:
    """Отобразить скрытую величину на заданное дискретное распределение.

    Ранг человека по скрытой величине определяет, в какую корзину он попадёт;
    размеры корзин заданы долями. Так маргинальное распределение получается
    точным, а монотонная связь со скрытой величиной сохраняется полностью —
    в отличие от обрезки, которая сминает хвост в одну точку.
    """
    n = len(latent)
    order = np.argsort(latent, kind="stable")
    edges = (np.cumsum(share / share.sum()) * n).astype(np.int64)
    out = np.empty(n, dtype=values.dtype)
    start = 0
    for i, end in enumerate(edges):
        out[order[start:int(end)]] = values[i]
        start = int(end)
    if start < n:
        out[order[start:]] = values[-1]
    return out


def _days_to_iso(days: np.ndarray) -> np.ndarray:
    """Дни от эпохи → ISO-строки. Считается только при записи колонки."""
    base = np.datetime64("1970-01-01")
    return (base + days.astype("timedelta64[D]")).astype("datetime64[D]").astype(str)


@dataclass
class Population:
    """Колоночное представление людей. Строки — люди, порядок стабилен."""

    n: int
    seed: int
    tree: org.OrgTree
    col: dict[str, np.ndarray] = field(default_factory=dict)
    truth: dict[str, np.ndarray] = field(default_factory=dict)
    book: NameBook = None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.col[name]

    def __contains__(self, name: str) -> bool:
        return name in self.col

    def get(self, name: str, default=None):
        return self.col.get(name, default)


def build(seed: int, n_target: int) -> Population:
    """Собрать популяцию под организационное дерево."""
    tree = org.build(seed, n_target)
    unit_of = org.assign_people(tree)
    n = len(unit_of)
    idx = np.arange(n)
    p = Population(n=n, seed=seed, tree=tree, book=NameBook(seed))
    c, truth = p.col, p.truth

    # ---------------------------------------------------------- скрытые факторы
    ability = normal(key64(seed, "lat.ability"), idx)
    potential = 0.55 * ability + 0.45 * normal(key64(seed, "lat.potential"), idx)
    engagement_z = 0.35 * ability + 0.65 * normal(key64(seed, "lat.engage"), idx)
    diligence = 0.30 * ability + 0.70 * normal(key64(seed, "lat.diligence"), idx)
    truth.update(ability=ability, potential=potential, engagement_z=engagement_z,
                 diligence=diligence)

    # ------------------------------------------------------------- организация
    c["unit_id"] = unit_of
    c["block_idx"] = tree.block[unit_of]
    c["tb_idx"] = tree.tb[unit_of]
    c["city_idx"] = tree.city[unit_of]

    # --------------------------------------------------------------- личность
    female = bernoulli(key64(seed, "sex"), idx, 0.62)   # в банке женщин больше
    c["female"] = female
    names = p.book.draw(idx, female)
    c["last_name"], c["first_name"] = names["last"], names["first"]
    c["middle_name"], c["full_name"] = names["middle"], names["full"]
    c["surname_index"] = names["surname_index"]

    # Возраст: корзина по целевым долям, затем равномерно внутри корзины.
    bucket = weighted(key64(seed, "age.bucket"), idx,
                      _cum(np.array([b[2] for b in AGE_BUCKETS])))
    lo = np.array([b[0] for b in AGE_BUCKETS])[bucket]
    hi = np.array([b[1] for b in AGE_BUCKETS])[bucket]
    age = lo + (cells(key64(seed, "age.in"), idx) % (hi - lo).astype(np.uint64)
                ).astype(np.int64)
    c["age"] = age
    birth_offset = integers(key64(seed, "birth.day"), idx, 0, 364)
    c["birth_days"] = AS_OF_DAYS - age * 365 - birth_offset
    c["birth_year"] = 1970 + (c["birth_days"] // 365)

    # ------------------------------------------------------------ стаж и найм
    # Опыт растёт с возрастом: доля трудоспособных лет, проведённых в работе.
    # Независимый от возраста опыт давал corr(возраст, грейд) ≈ 0,08 — в реальной
    # организации связь есть, просто она не единственная.
    max_exp = np.maximum(age - 18, 1).astype(np.float64)
    share_worked = np.clip(0.82 + 0.14 * normal(key64(seed, "exp.share"), idx),
                           0.25, 1.0)
    total_exp = np.maximum(max_exp * share_worked, 0.5)
    tenure = np.minimum(exponential(key64(seed, "exp.tenure"), idx, 5.0) + 0.15,
                        total_exp)
    c["total_exp_years"] = total_exp
    c["tenure_years"] = tenure
    c["hire_days"] = AS_OF_DAYS - (tenure * 365.25).astype(np.int64)

    # ------------------------------------------------------------------ грейд
    # Старшинство: способность, опыт и стаж. Отображается на целевую пирамиду.
    # Нагрузки подобраны так, чтобы corr(способность, грейд) ≈ 0,55, а
    # corr(возраст, грейд) ≈ 0,4: грейд несёт сигнал о силе, но не сводится
    # ни к ней, ни к выслуге. При corr 0,8 задача ранжирования вырождается.
    seniority = (0.42 * ability
                 + 0.55 * (np.log1p(total_exp) - np.log1p(total_exp).mean())
                 + 0.15 * (np.log1p(tenure) - np.log1p(tenure).mean())
                 + 0.55 * normal(key64(seed, "grade.noise"), idx))
    grade = _quantile_map(seniority, GRADE_LEVELS, GRADE_SHARE)
    c["grade_level"] = grade
    truth["seniority"] = seniority

    # ------------------------------------------------------- семья должностей
    block_names = [b for b, _ in dicts.BLOCKS]
    fam_names = sorted(dicts.JOB_FAMILIES)
    fam_index = {f: i for i, f in enumerate(fam_names)}
    fam_choice = np.zeros(n, dtype=np.int64)
    for bi, bname in enumerate(block_names):
        pool = [fam_index[f] for f in dicts.BLOCK_FAMILIES[bname]]
        mask = c["block_idx"] == bi
        if not mask.any():
            continue
        take = pick(key64(seed, "family", bi), idx[mask], len(pool))
        fam_choice[mask] = np.array(pool)[take]
    c["family_idx"] = fam_choice

    # Роль внутри семьи: чем выше грейд, тем старше роль.
    fam_sizes = np.array([len(dicts.JOB_FAMILIES[f]) for f in fam_names])
    span = fam_sizes[fam_choice]
    rank = np.clip((grade - 6) / 14.0, 0.0, 1.0)
    jitter = 0.12 * normal(key64(seed, "role.jitter"), idx)
    role = np.clip(((rank + jitter) * (span - 1)).round().astype(np.int64),
                   0, span - 1)
    c["role_idx"] = role
    c["family_names"] = np.array(fam_names, dtype=object)

    # ------------------------------------------------------- статус занятости
    # Уволенные — доля годовой текучести; дата увольнения в прошлом году.
    fired = bernoulli(key64(seed, "status.fired"), idx, ANNUAL_ATTRITION)
    fired &= tenure > 0.2
    c["fired"] = fired
    c["fired_days"] = np.where(
        fired, AS_OF_DAYS - integers(key64(seed, "fired.when"), idx, 1, 365), -1)
    c["status_idx"] = fired.astype(np.int64)          # 0 активный, 1 уволен
    c["form_idx"] = weighted(key64(seed, "form"), idx, _cum(
        np.array([w for _, w in dicts.FORM_WORK])))
    c["fact_flag"] = ((~fired) & (c["form_idx"] == 0)).astype(np.int64)
    c["work_mode_idx"] = weighted(key64(seed, "wmode"), idx,
                                  _cum(np.array([w for _, w in dicts.WORK_MODE])))
    c["family_status_idx"] = weighted(key64(seed, "fstatus"), idx, _cum(
        np.array([w for _, w in dicts.FAMILY_STATUS])))
    c["mobility_idx"] = weighted(key64(seed, "mobility"), idx, _cum(
        np.array([w for _, w in dicts.MOBILITY_STATUS])))

    # Дети: зависят от возраста и семейного положения — иначе двадцатилетние
    # холостяки получают троих детей и любая HR-агрегация выглядит абсурдно.
    child_p = np.clip((age - 22) / 60.0, 0.0, 0.55)
    child_p = np.where(c["family_status_idx"] == 0, child_p * 1.8, child_p * 0.5)
    kids = (bernoulli(key64(seed, "kid1"), idx, np.clip(child_p, 0, 0.9)).astype(int)
            + bernoulli(key64(seed, "kid2"), idx, np.clip(child_p * 0.55, 0, 0.7)).astype(int)
            + bernoulli(key64(seed, "kid3"), idx, np.clip(child_p * 0.12, 0, 0.3)).astype(int))
    c["children_qty"] = kids

    # Испытательный срок — настоящий признак, а не мёртвая колонка: три месяца.
    c["probation"] = (tenure < 0.25).astype(np.int64)
    c["intern"] = ((age < 25) & (grade <= 8) & (tenure < 1.5)).astype(np.int64)

    # ------------------------------------------------- срок в роли и повышения
    # Срок в роли не больше стажа; повышение — событие в истории.
    role_years = np.minimum(exponential(key64(seed, "role.years"), idx, 2.6) + 0.1,
                            tenure)
    c["role_years"] = role_years
    c["position_start_days"] = AS_OF_DAYS - (role_years * 365.25).astype(np.int64)
    c["promoted_last_2y"] = ((role_years < 2.0) & (tenure > role_years + 0.5)
                             ).astype(np.int64)

    # ------------------------------------------------------------ руководство
    # Руководитель — тот, кого назначили головой подразделения (см. ниже).
    c["is_head"] = np.zeros(n, dtype=np.int64)
    c["head_unit"] = np.full(n, -1, dtype=np.int64)
    _appoint_heads(p)

    # ----------------------------------------------------------- результативность
    # Скрытая результативность — вход порядковой модели оценок.
    perf_latent = (0.62 * ability + 0.22 * diligence
                   + 0.35 * normal(key64(seed, "perf.noise"), idx))
    truth["perf_latent"] = perf_latent
    marks = _ordinal_marks(perf_latent, seed, idx)
    c["mark_perf"] = marks                                   # 8 кварталов, 0..4
    truth["perf_avg_5"] = 5.0 - marks[:, -4:].mean(axis=1)   # A=5 … E=1

    # ------------------------------------------------------------- компетенции
    comps = np.empty((n, len(dicts.COMPETENCIES)), dtype=np.float32)
    for k in range(len(dicts.COMPETENCIES)):
        specific = normal(key64(seed, "comp", k), idx)
        raw = 3.05 + 0.62 * ability + 0.42 * specific
        if dicts.COMPETENCIES[k] in ("Развитие команды", "Лидерство"):
            raw += 0.35 * (c["is_head"] == 1)   # управленческие растут у руководителей
        comps[:, k] = np.clip(np.round(raw * 10) / 10, 1.0, 5.0)
    c["competencies"] = comps
    truth["competency_avg"] = comps.mean(axis=1)

    # ------------------------------------------------------------ вовлечённость
    engagement = np.clip(np.round(62 + 9.5 * engagement_z + 4.0 * (~fired)), 5, 100)
    c["engagement"] = engagement.astype(np.int64)

    # ------------------------------------------------------------- риск оттока
    risk_latent = (-0.42 * ability - 0.030 * (engagement - 65)
                   + 0.28 * np.log1p(role_years)
                   - 0.22 * c["promoted_last_2y"]
                   + 0.18 * (c["form_idx"] > 0)
                   + 0.45 * normal(key64(seed, "risk.noise"), idx))
    risk = _quantile_map(risk_latent, np.array([0, 1, 2]),
                         np.array([0.58, 0.30, 0.12]))
    c["attrition_risk_idx"] = risk        # 0 низкий, 1 средний, 2 высокий
    truth["attrition_risk_latent"] = risk_latent

    # ------------------------------------------------------ сила профиля и HR-метки
    scored = ~bernoulli(key64(seed, "star.scored"), idx, UNSCORED_RATE)
    star = np.clip(np.round(52 + 16 * ability + 8 * normal(key64(seed, "star.n"), idx)),
                   1, 100).astype(np.int64)
    c["star_index"] = np.where(scored, star, -1)     # -1 → NULL при записи
    c["has_embedding"] = bernoulli(key64(seed, "emb"), idx, 0.74)

    # Кадровый резерв — управленческое решение прошлого года, а не расчёт.
    pool_score = 0.6 * potential + 0.4 * ability + 0.5 * normal(key64(seed, "pool"), idx)
    c["talent_pool"] = (pool_score > np.quantile(pool_score, 0.92)).astype(np.int64)
    c["career_status_idx"] = _career_status(potential, role_years, c["talent_pool"])

    # ------------------------------------------------------------ идентификаторы
    c["employee_id"] = _unique_tab_numbers(seed, idx)
    c["company_idx"] = weighted(key64(seed, "company"), idx,
                                _cum(np.array(dicts.COMPANY_WEIGHTS)))
    c["position_id"] = 100_000 + idx                  # позиция уникальна на человека
    return p


def _appoint_heads(p: Population) -> None:
    """Назначить руководителей: голова подразделения — сотрудник из него.

    Берётся человек с наибольшим грейдом среди сотрудников листа; для узлов выше
    — руководитель самого крупного дочернего подразделения. Так строится
    настоящая цепочка подчинения, а не набор выдуманных строк.
    """
    tree, c = p.tree, p.col
    unit_of, grade = c["unit_id"], c["grade_level"]
    order = np.lexsort((-grade, unit_of))
    sorted_units = unit_of[order]
    first = np.r_[True, sorted_units[1:] != sorted_units[:-1]]
    head_rows = order[first]
    head_units = unit_of[head_rows]

    tree.head_person[head_units] = head_rows
    c["is_head"][head_rows] = 1
    c["head_unit"][head_rows] = head_units

    # Вверх по дереву: руководителем узла становится глава первого дочернего
    # подразделения, у которого руководитель уже есть.
    for lvl in range(int(tree.level.max()) - 1, 0, -1):
        nodes = np.flatnonzero((tree.level == lvl) & (tree.head_person < 0))
        for node in nodes:
            children = np.flatnonzero(tree.parent == node)
            heads = tree.head_person[children]
            heads = heads[heads >= 0]
            if len(heads):
                best = heads[np.argmax(grade[heads])]
                tree.head_person[node] = best
                c["head_unit"][best] = node          # верхний узел важнее листа


def _ordinal_marks(latent: np.ndarray, seed: int, idx: np.ndarray,
                   quarters: int = 8) -> np.ndarray:
    """Оценки A–E за ``quarters`` кварталов порядковой моделью.

    Пороги режут скрытую результативность так, чтобы доли совпали с политикой
    принудительного ранжирования. Между кварталами добавляется свой шум —
    иначе у человека все восемь оценок одинаковы и тренд не определён.
    """
    n = len(latent)
    out = np.empty((n, quarters), dtype=np.int8)
    # Порядок важен: индекс 0 — это «A», лучшая оценка, а она достаётся
    # НАИБОЛЬШЕЙ скрытой результативности. Прямое отображение перевернуло бы
    # шкалу и дало отрицательную связь оценки с грейдом.
    share = np.array(dicts.ESTIMATION_TARGET)[::-1]
    marks = np.arange(len(share))[::-1]
    # Устойчивая склонность оценивающего: у одного руководителя оценки мягче.
    # Без неё усреднение восьми кварталов вычищает шум и оценка становится
    # почти безошибочной копией способности (corr 0,86) — задача вырождается.
    rater = 0.55 * normal(key64(seed, "mark.rater"), idx)
    for q in range(quarters):
        drift = 0.70 * normal(key64(seed, "mark", q), idx)
        out[:, q] = _quantile_map(latent + rater + drift, marks, share)
    return out


def _unique_tab_numbers(seed: int, idx: np.ndarray) -> np.ndarray:
    """Табельные номера: случайные на вид, но гарантированно различные.

    Прежде номер был просто ``1_000_000 + hash % 8_999_999``. На трёх тысячах
    человек это выглядело безупречно, а на 294 000 парадокс дней рождения даёт
    ожидаемых совпадений 294000² / (2 · 8999999) ≈ 4800 — измерено 4835 лишних
    строк на 4779 значений, то есть 1,6% людей делили номер с кем-то ещё.

    Тихо это не проходило нигде, где номер используется как ключ:

    * ``sim/emulator/identity.py`` строит ``{employee_id: строка}``, и у двух
      людей с одним номером остаётся один разрешающий скоуп на двоих — победил
      тот, кто был позже в массиве. Один человек молча получал права другого.
    * Гейт W7 сверяет ФИО преемника по его номеру и падал ровно на этих 1,6%;
      до того как преемники стали настоящими, номер там был мусорным, и
      проверять было нечего.
    * Промпт требует от агента ссылаться на проверяемый идентификатор. Номер,
      указывающий на двух разных людей, этому требованию не отвечает.

    Починка: развести совпадения, а не расширять диапазон. Значения сортируются,
    делаются строго возрастающими минимальным сдвигом вверх и возвращаются на
    свои места. Каждый номер сдвигается на величину своего кластера совпадений —
    единицы, — поэтому «случайный» вид сохраняется, а порядок ``idx`` в номере
    по-прежнему не читается: сортировка идёт по хешу, не по строке.
    """
    raw = (1_000_000 + (cells(key64(seed, "tab"), idx)
                        % np.uint64(8_999_999)).astype(np.int64))
    order = np.argsort(raw, kind="stable")
    ranks = np.arange(len(raw), dtype=np.int64)
    # Классический приём: вычесть ранг, взять бегущий максимум, прибавить ранг
    # обратно. Результат строго возрастает, а значит различен, и каждый элемент
    # не меньше исходного.
    bumped = np.maximum.accumulate(raw[order] - ranks) + ranks
    out = np.empty_like(raw)
    out[order] = bumped
    return out


def _career_status(potential: np.ndarray, role_years: np.ndarray,
                   pool: np.ndarray) -> np.ndarray:
    """Карьерный статус: наблюдаемая метка, согласованная с потенциалом."""
    status = np.full(len(potential), 1, dtype=np.int64)      # развивается в роли
    status = np.where(role_years < 1.0, 0, status)           # новичок в роли
    status = np.where((potential > 0.85) & (role_years >= 1.0), 3, status)
    status = np.where((potential < -0.2) & (role_years > 4), 2, status)
    return np.where(pool == 1, 4, status)


def position_name(p: Population) -> np.ndarray:
    """Название должности по семье и роли."""
    fam_names = list(p["family_names"])
    table = [dicts.JOB_FAMILIES[f] for f in fam_names]
    return np.array([table[f][r] for f, r in zip(p["family_idx"], p["role_idx"])],
                    dtype=object)
