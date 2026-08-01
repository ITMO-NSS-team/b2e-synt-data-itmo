"""Блоки параллельных массивов: плоское хранение + позиционное выравнивание.

Инвариант блока
---------------
``estimation.year[i]`` относится к ``estimation.quarter[i]`` и
``estimation.performance[i]``. В прежнем корпусе он держался только у явно
запрограммированных блоков: у неявных групп (``educational_institution_name`` и
соседи) длины массивов совпадали на 4 строках из 3000, то есть «где учился и по
какой специальности» не имело верного ответа.

Здесь блок — это **плоские массивы полей плюс смещения по людям**. Массив поля
физически один; нарезка на людей общая для всех полей блока. Рассогласовать
длины невозможно: они берутся из одного ``counts``.

Плоское хранение выбрано ещё и по памяти: 300 000 человек × 25 записей — это
7,5 млн записей, и держать их списком словарей нельзя при 3 ГБ ОЗУ.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from . import dicts
from .population import AS_OF_DAYS, Population, _days_to_iso
from .rng import bernoulli, cells, integers, key64, normal, pick, unit, weighted


@dataclass
class Block:
    """Один блок: поля в плоском виде и границы записей по людям."""

    counts: np.ndarray               # сколько записей у каждого человека
    offsets: np.ndarray              # counts.cumsum() со сдвигом
    fields: dict[str, np.ndarray]    # имя поля → плоский массив длины total

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def lists(self, field: str, rows: np.ndarray) -> list:
        """Массивы значений поля для указанных людей."""
        flat = self.fields[field]
        return [flat[self.offsets[r]:self.offsets[r] + self.counts[r]].tolist()
                for r in rows]

    def flat_rows(self) -> np.ndarray:
        """Номер человека для каждой плоской записи — нужен для развёрток."""
        return np.repeat(np.arange(len(self.counts)), self.counts)


def _frame(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    return offsets, np.repeat(np.arange(len(counts)), counts)


def estimation(p: Population) -> Block:
    """Оценки по кварталам: 1–4 квартальные, 5 — годовая. Коды 6–7 отменены."""
    seed, n = p.seed, p.n
    marks = p["mark_perf"]                       # (n, 8) от старых к новым
    periods = [(2024, 3), (2024, 4), (2025, 1), (2025, 2),
               (2025, 3), (2025, 4), (2026, 1), (2026, 2)]
    # Пропуски: человек мог быть в отпуске или не иметь оценки за квартал.
    keep = np.ones((n, len(periods)), dtype=bool)
    for q in range(len(periods)):
        keep[:, q] = ~bernoulli(key64(seed, "est.skip", q), np.arange(n), 0.07)
    # Оценки до найма не существуют.
    tenure_q = (p["tenure_years"] * 4).astype(np.int64)
    for q in range(len(periods)):
        keep[:, q] &= tenure_q >= (len(periods) - q)

    counts = keep.sum(axis=1).astype(np.int64)
    offsets, rows = _frame(counts)
    flat_q = np.tile(np.arange(len(periods)), n).reshape(n, -1)[keep]
    year = np.array([y for y, _ in periods])[flat_q]
    quarter = np.array([q for _, q in periods])[flat_q]
    mark_idx = marks[keep]
    letters = np.array(dicts.ESTIMATION_MARKS)
    values_idx = (mark_idx + integers(key64(seed, "est.values"),
                                      np.arange(len(mark_idx)), -1, 1)).clip(0, 4)
    perf = np.array([f"{a} {b}" for a, b in
                     zip(letters[mark_idx], letters[values_idx])], dtype=object)
    standards = np.where(bernoulli(key64(seed, "est.std"),
                                   np.arange(len(mark_idx)), 0.88),
                         "Соблюдает", "Требует внимания").astype(object)
    return Block(counts, offsets, {
        "year": year, "quarter": quarter, "performance": perf,
        "standards": standards, "mark_index": mark_idx,
    })


def goals(p: Population) -> Block:
    """Цели: прогресс связан с результативностью, а не случаен."""
    seed, n = p.seed, p.n
    idx = np.arange(n)
    counts = integers(key64(seed, "goals.n"), idx, 0, 6)
    counts = np.where(p["fired"] == 1, 0, counts)
    offsets, rows = _frame(counts)
    m = int(counts.sum())
    flat = np.arange(m)
    base = p.truth["perf_latent"][rows]
    progress = np.clip(0.62 + 0.16 * base + 0.22 * normal(key64(seed, "goal.p"), flat),
                       0.0, 1.25).round(4)
    return Block(counts, offsets, {
        "title": np.array(dicts.GOAL_TITLES, dtype=object)[
            pick(key64(seed, "goal.t"), flat, len(dicts.GOAL_TITLES))],
        "type": np.array(dicts.GOAL_TYPES, dtype=object)[
            pick(key64(seed, "goal.ty"), flat, len(dicts.GOAL_TYPES))],
        "quarter": integers(key64(seed, "goal.q"), flat, 1, 4),
        "progress": progress,
        "tags": np.array(["стратегия", "эффективность", "клиент", "команда",
                          "качество"], dtype=object)[
            pick(key64(seed, "goal.tag"), flat, 5)],
        "risk": np.where(progress < 0.5, "Под риском", "В графике").astype(object),
    })


def learning(p: Population) -> Block:
    """Назначенное обучение: обязательные курсы у всех, добровольные — по интересу."""
    seed, n = p.seed, p.n
    idx = np.arange(n)
    # Дисциплинированные проходят больше и в срок — это и делает витрину
    # обучения сигналом, а не украшением.
    base = 3 + (2.5 * (p.truth["diligence"] + 1.2)).clip(0, 9).astype(np.int64)
    counts = np.where(p["fired"] == 1, (base // 2), base)
    offsets, rows = _frame(counts)
    m = int(counts.sum())
    flat = np.arange(m)

    course_idx = pick(key64(seed, "lrn.c"), flat, len(dicts.COURSES))
    names = np.array([c[0] for c in dicts.COURSES], dtype=object)[course_idx]
    kinds = np.array([c[1] for c in dicts.COURSES], dtype=object)[course_idx]
    hours = np.array([c[2] for c in dicts.COURSES])[course_idx]

    assigned = AS_OF_DAYS - integers(key64(seed, "lrn.a"), flat, 20, 900)
    deadline = assigned + hours * 3 + integers(key64(seed, "lrn.d"), flat, 14, 120)
    diligence = p.truth["diligence"][rows]
    done_p = np.clip(0.55 + 0.18 * diligence, 0.15, 0.95)
    done = bernoulli(key64(seed, "lrn.done"), flat, done_p)
    overdue = (~done) & (deadline < AS_OF_DAYS)
    cancelled = bernoulli(key64(seed, "lrn.cancel"), flat, 0.05)
    status = np.where(cancelled, "CANCELLED",
                      np.where(done, "COMPLETED",
                               np.where(overdue, "OVERDUE", "ACTIVE"))).astype(object)
    completed = np.where(done & ~cancelled,
                         assigned + integers(key64(seed, "lrn.cd"), flat, 3, 200), -1)
    return Block(counts, offsets, {
        "assignment_id": 100000 + flat,
        "assignment_status": status,
        "assignment_type": kinds,
        "item_name": names,
        "item_type": np.array(["Курс", "Тест", "Вебинар", "Трек"], dtype=object)[
            pick(key64(seed, "lrn.t"), flat, 4)],
        "item_format_name": np.array(["Онлайн", "Очно", "Смешанный"], dtype=object)[
            pick(key64(seed, "lrn.f"), flat, 3)],
        "item_duration": hours,
        "assigned_days": assigned,
        "deadline_days": deadline,
        "completed_days": completed,
    })


def education(p: Population) -> Block:
    """Академическое образование: вуз, уровень, специальность, годы — согласованно.

    Специальность выбирается из профиля вуза: выпускник МФТИ не выходит с
    «Юриспруденцией». Год окончания согласован с возрастом.
    """
    seed, n = p.seed, p.n
    idx = np.arange(n)
    # Число образований: у большинства одно, у части второе и MBA.
    counts = (1 + bernoulli(key64(seed, "edu.2"), idx, 0.34).astype(np.int64)
              + bernoulli(key64(seed, "edu.3"), idx, 0.07).astype(np.int64))
    counts = np.where(p["age"] < 21, 0, counts)
    offsets, rows = _frame(counts)
    m = int(counts.sum())
    flat = np.arange(m)

    uni_idx = weighted(key64(seed, "edu.u"), flat,
                       np.cumsum(1.0 / np.arange(1, len(dicts.UNIVERSITIES) + 1) ** 0.9
                                 / (1.0 / np.arange(1, len(dicts.UNIVERSITIES) + 1) ** 0.9).sum()))
    uni_names = np.array([u[0] for u in dicts.UNIVERSITIES], dtype=object)[uni_idx]
    uni_city = np.array([u[1] for u in dicts.UNIVERSITIES], dtype=object)[uni_idx]
    profiles = np.array([u[2] for u in dicts.UNIVERSITIES])[uni_idx]

    spec = np.empty(m, dtype=object)
    for prof, options in dicts.SPECIALTIES_BY_PROFILE.items():
        mask = profiles == prof
        if mask.any():
            spec[mask] = np.array(options, dtype=object)[
                pick(key64(seed, "edu.s", prof), flat[mask], len(options))]

    level_idx = weighted(key64(seed, "edu.l"), flat, np.cumsum(
        np.array([w for _, w in dicts.EDUCATION_LEVELS])
        / sum(w for _, w in dicts.EDUCATION_LEVELS)))
    level = np.array([l for l, _ in dicts.EDUCATION_LEVELS], dtype=object)[level_idx]

    age = p["age"][rows]
    grad_age = np.clip(21 + integers(key64(seed, "edu.ga"), flat, 0, 6), 21, age)
    end_year = (2026 - (age - grad_age)).astype(np.int64)
    return Block(counts, offsets, {
        "institution": uni_names, "institution_city": uni_city,
        "speciality": spec, "level": level,
        "end_year": end_year, "end_month": integers(key64(seed, "edu.m"), flat, 1, 12),
        "start_year": end_year - np.where(level_idx <= 1, 4, 5),
        "diploma_id": 5_000_000 + flat,
    })


def absence(p: Population) -> Block:
    """Отсутствия: отпуска, больничные, командировки за последние два года."""
    seed, n = p.seed, p.n
    idx = np.arange(n)
    counts = integers(key64(seed, "abs.n"), idx, 0, 7)
    offsets, rows = _frame(counts)
    m = int(counts.sum())
    flat = np.arange(m)
    kind = pick(key64(seed, "abs.k"), flat, len(dicts.ABSENCE_KINDS))
    names = np.array([k[0] for k in dicts.ABSENCE_KINDS], dtype=object)[kind]
    groups = np.array([k[1] for k in dicts.ABSENCE_KINDS], dtype=object)[kind]
    codes = np.array([k[2] for k in dicts.ABSENCE_KINDS])[kind]
    typical = np.array([k[3] for k in dicts.ABSENCE_KINDS])[kind]
    start = AS_OF_DAYS - integers(key64(seed, "abs.s"), flat, 5, 730)
    length = np.maximum(1, (typical * (0.5 + unit(key64(seed, "abs.l"), flat))
                            ).astype(np.int64))
    return Block(counts, offsets, {
        "name": names, "group_name": groups, "code": codes,
        "group": np.array([1, 1, 1, 2, 2, 3, 4, 5, 5, 6])[kind],
        "start_days": start, "end_days": start + length, "days": length,
        "blocked": (groups == "Больничный").astype(np.int64),
    })


def successors(p: Population, rng_ns: str = "succ") -> Block:
    """Преемники: реальные сотрудники того же блока, а не выдуманные строки.

    Ссылочная целостность здесь — не педантизм: «кто может заменить руководителя»
    относится к людям, которых агент должен уметь найти в других витринах.
    """
    seed, n = p.seed, p.n
    idx = np.arange(n)
    is_head = p["is_head"] == 1
    counts = np.where(is_head, integers(key64(seed, rng_ns, "n"), idx, 0, 3), 0)
    # У части руководителей преемников нет — это осмысленный ответ «резерв пуст».
    offsets, rows = _frame(counts)
    m = int(counts.sum())
    flat = np.arange(m)

    # Кандидат берётся из того же блока и грейдом не ниже, чем на 2 ниже.
    block_of = p["block_idx"]
    order = np.argsort(block_of, kind="stable")
    block_sorted = block_of[order]
    starts = np.searchsorted(block_sorted, block_of[rows], side="left")
    ends = np.searchsorted(block_sorted, block_of[rows], side="right")
    span = np.maximum(ends - starts, 1)
    choice = order[starts + (cells(key64(seed, rng_ns, "pick"), flat)
                             % span.astype(np.uint64)).astype(np.int64)]
    return Block(counts, offsets, {
        "row": choice,
        "status": np.array(["Готов сейчас", "Готов через 1–2 года",
                            "На рассмотрении", "Утверждён"], dtype=object)[
            pick(key64(seed, rng_ns, "st"), flat, 4)],
        "appoint_days": AS_OF_DAYS - integers(key64(seed, rng_ns, "d"), flat, 30, 1200),
    })


def career(p: Population) -> Block:
    """Трудовая история: внешние места работы до найма плюс внутренние переходы."""
    seed, n = p.seed, p.n
    idx = np.arange(n)
    external_years = np.maximum(p["total_exp_years"] - p["tenure_years"], 0)
    counts = np.clip((external_years / 3.2).astype(np.int64), 0, 6)
    offsets, rows = _frame(counts)
    m = int(counts.sum())
    flat = np.arange(m)
    end = (AS_OF_DAYS - (p["tenure_years"][rows] * 365.25).astype(np.int64)
           - integers(key64(seed, "car.gap"), flat, 0, 400)
           - (np.arange(m) - offsets[rows]) * 900)
    return Block(counts, offsets, {
        "company": np.array(dicts.EXT_COMPANIES, dtype=object)[
            pick(key64(seed, "car.c"), flat, len(dicts.EXT_COMPANIES))],
        "position": np.array(sum(dicts.JOB_FAMILIES.values(), []), dtype=object)[
            pick(key64(seed, "car.p"), flat,
                 len(sum(dicts.JOB_FAMILIES.values(), [])))],
        "end_days": end,
        "start_days": end - integers(key64(seed, "car.len"), flat, 200, 1800),
    })


def achievements(p: Population) -> Block:
    seed, n = p.seed, p.n
    idx = np.arange(n)
    strong = p.truth["perf_latent"] > 0.8
    counts = np.where(strong, integers(key64(seed, "ach.n"), idx, 0, 3), 0)
    offsets, rows = _frame(counts)
    m = int(counts.sum())
    flat = np.arange(m)
    return Block(counts, offsets, {
        "name": np.array(dicts.ACHIEVEMENTS, dtype=object)[
            pick(key64(seed, "ach.a"), flat, len(dicts.ACHIEVEMENTS))],
        "description": np.array(["Награда по итогам внутреннего конкурса"],
                                dtype=object)[np.zeros(m, dtype=np.int64)],
        "period": np.array(["2024", "2025", "2026H1"], dtype=object)[
            pick(key64(seed, "ach.p"), flat, 3)],
    })


BUILDERS = {
    "estimation": estimation, "goals": goals, "learning": learning,
    "education": education, "absence": absence, "successors": successors,
    "career": career, "achievements": achievements,
}


def build_all(p: Population) -> dict[str, Block]:
    return {name: fn(p) for name, fn in BUILDERS.items()}
