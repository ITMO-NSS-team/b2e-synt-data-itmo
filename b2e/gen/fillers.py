"""Процедурные значения для колонок, не входящих в смысловое ядро.

Зачем длинный хвост вообще заполняется
--------------------------------------
В каталоге 4 599 колонок. Большинство к конкретному вопросу отношения не имеет,
но агент обязан **искать** нужную колонку в реалистично широком каталоге —
именно на этом ломаются боевые агенты, и именно это симулятор должен measure.
Пустой хвост сделал бы поиск тривиальным.

Почему процедурно, а не на диск
-------------------------------
300 000 строк × 3 800 хвостовых колонок — это 1,14 млрд ячеек. Материализовать
их нельзя ни по месту, ни по времени. Значение вычисляется на чтении из
координаты ``(seed, витрина, колонка, номер строки)`` и потому одинаково в любом
процессе и при любом порядке обращения.

Чем это отличается от «мусора»
------------------------------
Правило выбирается **по имени** раньше, чем по типу: ``*_email`` даёт почту,
``*_dt`` — дату, ``*_name`` — название из словаря. Строка вида «промежуточный
institute_name 61» из прежнего корпуса — это отсутствующее правило, а не
неизбежность; здесь такие имена перекрыты словарями.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from . import dicts
from .population import AS_OF_DAYS, _days_to_iso
from .rng import bernoulli, cells, integers, key64, normal, pick, unit, weighted

#: Доля NULL у nullable-колонок: без неё нечего проверять ни в ``IS NULL``,
#: ни в каверзе «сортировка без nulls: last».
NULL_RATE = 0.07
EMBEDDING_DIM = 384
MAX_ARRAY = 5

EARLIEST_DAYS = AS_OF_DAYS - 365 * 16


def _hexid(base, rows: np.ndarray) -> np.ndarray:
    """UUID-подобный идентификатор, детерминированный по координате."""
    a = cells(base, rows)
    b = cells(base ^ np.uint64(0x5DEECE66D), rows)
    return np.array([f"{int(x):016x}-{int(y):04x}-4{int(y) >> 16 & 0xfff:03x}-"
                     f"a{int(y) >> 28 & 0xfff:03x}-{int(y) >> 8 & 0xffffffffffff:012x}"
                     for x, y in zip(a, b)], dtype=object)


def _dates(base, rows: np.ndarray, lo: int = EARLIEST_DAYS,
           hi: int = AS_OF_DAYS) -> np.ndarray:
    return _days_to_iso(integers(base, rows, lo, hi))


def _from(pool: list, base, rows: np.ndarray) -> np.ndarray:
    arr = np.array(pool, dtype=object)
    return arr[pick(base, rows, len(arr))]


#: Правила по имени. Первое совпадение выигрывает, поэтому точные имена раньше.
_RULES: list[tuple[str, str]] = [
    (r"^person_id$|_person_id$|^id$|_uuid$", "uuid"),
    (r"^employee_id$|_employee_id$|^tn$|_tab_num$", "tab"),
    (r"^unit_id$|_unit_id$|^org_unit", "unitid"),
    (r"_email$|^email", "email"),
    (r"phone", "phone"),
    (r"^full_name$|_full_name$|^fio", "fullname"),
    (r"^first_name$|_first_name$", "firstname"),
    (r"^last_name$|_last_name$|^surname$", "lastname"),
    (r"^mid_name$|_mid_name$|patronymic", "midname"),
    (r"institut|universit|educational_institution|^vuz", "university"),
    (r"speciality|specialty|faculty|^napravlenie", "speciality"),
    (r"education_type|education_level|degree|qualification", "edulevel"),
    (r"course|item_name$|playlist_name$|program_name$|^training", "course"),
    (r"competenc", "competency"),
    (r"^skill|skill_name$|_skills?$", "skill"),
    (r"city|town|^gosb_name$|^vsp_name$|locality", "city"),
    (r"region", "region"),
    (r"address|adres", "address"),
    (r"^block_name$|^fblock_name$|^blok", "block"),
    (r"unit_name|department|division|podrazdel", "unitname"),
    (r"position_name|^job_title|dolzhnost", "position"),
    (r"^company$|^org$|labor_book_company|current_company|employer", "company"),
    (r"^grade$|_grade$", "gradestr"),
    (r"grade_level|_grade_num$", "gradenum"),
    (r"^gender$|^sex$|^pol$", "gender"),
    (r"^generation$", "generation"),
    (r"^age$|_age$", "age"),
    (r"family_status|marital", "family"),
    (r"^work_mode$|^format_raboty$", "workmode"),
    (r"^form_work$", "formwork"),
    (r"status$|_state$|^state$", "status"),
    (r"^estimation|_mark$|performance_mark", "mark"),
    (r"goal_title|^goals?_name$", "goaltitle"),
    (r"goal_type", "goaltype"),
    (r"absence_name|^otsutstvie", "absencename"),
    (r"absence_group", "absencegroup"),
    (r"achievement|award|nagrada", "achievement"),
    (r"telegram|social|^vk_|^github", "handle"),
    (r"^is_|^has_|_flag$|^flag_|_indicator$|_priznak$", "flag"),
    (r"_pct$|_percent$|_share$|_ratio$|_rate$|completion|utilization", "ratio"),
    (r"_score$|_index$|_avg$|_mean$|_level$|_rating$", "score"),
    (r"_qty$|_count$|_cnt$|_num$|^num_|_amount$|_total$", "count"),
    (r"_days$|_day$|_duration$", "days"),
    (r"_year$|^year$", "year"),
    (r"_quarter$|^quarter$", "quarter"),
    (r"_month$|^month$", "month"),
    (r"_dt$|_date$|^date_|birthday|^dt_", "date"),
    (r"_at$|_time$|_ts$|timestamp", "datetime"),
    (r"_id$|_key$|^key$|_code$|^code$", "code"),
    (r"comment|description|note|text|reason|feedback", "text"),
    (r"_name$|^name$|^title$", "genericname"),
]
_COMPILED = [(re.compile(p), kind) for p, kind in _RULES]


@lru_cache(maxsize=16384)
def rule_for(name: str) -> str | None:
    tail = name.rsplit(".", 1)[-1]
    for pattern, kind in _COMPILED:
        if pattern.search(name) or pattern.search(tail):
            return kind
    return None


def is_embedding(name: str) -> bool:
    return bool(re.search(r"emb$|embs$|embedding|_emb_|vector$", name))


_POOLS = {
    "university": [u[0] for u in dicts.UNIVERSITIES],
    "speciality": sorted({s for v in dicts.SPECIALTIES_BY_PROFILE.values() for s in v}),
    "edulevel": [e[0] for e in dicts.EDUCATION_LEVELS],
    "course": [c[0] for c in dicts.COURSES],
    "competency": dicts.COMPETENCIES,
    "skill": sorted({s for v in dicts.SKILLS_BY_FAMILY.values() for s in v}),
    "city": [c[0] for c in dicts.CITIES],
    "region": dicts.REGION_CODES,
    "block": [b[0] for b in dicts.BLOCKS],
    "position": sorted({r for v in dicts.JOB_FAMILIES.values() for r in v}),
    "company": dicts.EXT_COMPANIES,
    "gender": ["М", "Ж"],
    "generation": ["Беби-бумеры", "X", "Y", "Z"],
    "family": [f[0] for f in dicts.FAMILY_STATUS],
    "workmode": [w[0] for w in dicts.WORK_MODE],
    "formwork": [f[0] for f in dicts.FORM_WORK],
    "status": ["активный", "уволен", "в отпуске", "на испытательном сроке"],
    "mark": [f"{a} {b}" for a in dicts.ESTIMATION_MARKS for b in dicts.ESTIMATION_MARKS],
    "goaltitle": dicts.GOAL_TITLES,
    "goaltype": dicts.GOAL_TYPES,
    "absencename": [a[0] for a in dicts.ABSENCE_KINDS],
    "absencegroup": sorted({a[1] for a in dicts.ABSENCE_KINDS}),
    "achievement": dicts.ACHIEVEMENTS,
    "firstname": dicts.MALE_NAMES + dicts.FEMALE_NAMES,
    "lastname": [r for r in dicts.SURNAME_ROOTS if r.endswith(("ов", "ев", "ин"))],
    "midname": sorted({p for n in dicts.MALE_NAMES
                       for p in dicts.PATRONYMIC_IRREGULAR.get(n, (n + "ович",
                                                                   n + "овна"))}),
}


def scalar(name: str, ch, base, rows: np.ndarray, names_book=None) -> np.ndarray:
    """Скалярное значение колонки по имени и типу."""
    kind = rule_for(name)
    pool = _POOLS.get(kind or "")
    if pool is not None:
        return _from(pool, base, rows)
    if kind == "uuid":
        return _hexid(base, rows)
    if kind == "tab":
        return (1_000_000 + integers(base, rows, 0, 8_999_999)).astype(object)
    if kind == "unitid":
        return integers(base, rows, 1000, 999_999)
    if kind == "email":
        return np.array([f"user{int(x) % 900000 + 10000}@sber.example"
                         for x in cells(base, rows)], dtype=object)
    if kind == "phone":
        return np.array([f"+79{int(x) % 900000000 + 10000000:09d}"
                         for x in cells(base, rows)], dtype=object)
    if kind == "fullname" and names_book is not None:
        female = bernoulli(base, rows, 0.6)
        return names_book.draw(rows, female)["full"]
    if kind == "address":
        return np.array([f"{c}, ул. {s}, д. {int(n) % 90 + 1}"
                         for c, s, n in zip(_from(_POOLS["city"], base, rows),
                                            _from(["Ленина", "Гагарина", "Мира",
                                                   "Советская", "Пушкина",
                                                   "Молодёжная", "Садовая",
                                                   "Вавилова", "Кутузовский"],
                                                  base ^ np.uint64(7), rows),
                                            cells(base, rows))], dtype=object)
    if kind == "handle":
        return np.array([f"@user{int(x) % 900000 + 1000}" for x in cells(base, rows)],
                        dtype=object)
    if kind == "gradestr":
        return np.array([f"G{g}" for g in integers(base, rows, 6, 20)], dtype=object)
    if kind == "gradenum":
        return integers(base, rows, 6, 20)
    if kind == "age":
        return integers(base, rows, 20, 63)
    if kind == "flag":
        return integers(base, rows, 0, 1)
    if kind == "ratio":
        return np.round(unit(base, rows), 4)
    if kind == "score":
        return np.round(1 + 4 * unit(base, rows), 2)
    if kind == "count":
        return integers(base, rows, 0, 60)
    if kind == "days":
        return integers(base, rows, 0, 365)
    if kind == "year":
        return integers(base, rows, 2015, 2026)
    if kind == "quarter":
        return integers(base, rows, 1, 4)
    if kind == "month":
        return integers(base, rows, 1, 12)
    if kind == "date":
        return _dates(base, rows)
    if kind == "datetime":
        d = _dates(base, rows)
        h = integers(base ^ np.uint64(3), rows, 0, 23)
        m = integers(base ^ np.uint64(5), rows, 0, 59)
        return np.array([f"{a} {b:02d}:{c:02d}:00" for a, b, c in zip(d, h, m)],
                        dtype=object)
    if kind == "code":
        return np.array([f"{int(x) % 900000 + 100000}" for x in cells(base, rows)],
                        dtype=object)
    if kind == "text":
        return _from(["Без замечаний", "Требует уточнения", "Согласовано",
                      "На рассмотрении", "Комментарий не заполнен",
                      "Уточнить у руководителя"], base, rows)
    if kind == "genericname":
        stem = name.rsplit(".", 1)[-1].replace("_", " ")
        return np.array([f"{w} {stem}" for w in _from(dicts.GENERIC_WORDS, base, rows)],
                        dtype=object)
    return by_type(ch, base, rows)


def by_type(ch, base, rows: np.ndarray) -> np.ndarray:
    """Последняя линия: значение по типу ClickHouse."""
    kind = getattr(ch, "kind", "string")
    if kind == "uuid":
        return _hexid(base, rows)
    if kind == "bool":
        return bernoulli(base, rows, 0.5)
    if kind == "date":
        return _dates(base, rows)
    if kind == "datetime":
        d = _dates(base, rows)
        return np.array([f"{x} 12:00:00" for x in d], dtype=object)
    if kind == "int":
        hi = 5 if getattr(ch, "base", "") in ("UInt8", "Int8") else 1000
        return integers(base, rows, 0, hi)
    if kind in ("float", "decimal"):
        return np.round(100 * unit(base, rows), 4)
    if kind == "map":
        keys = _from(dicts.GENERIC_WORDS, base, rows)
        return np.array([{k: int(v % 2)} for k, v in zip(keys, cells(base, rows))],
                        dtype=object)
    if kind == "tuple":
        return np.array([[dicts.COMPANIES[int(x) % len(dicts.COMPANIES)],
                          str(1_000_000 + int(x) % 8_999_999)]
                         for x in cells(base, rows)], dtype=object)
    return _from(dicts.GENERIC_WORDS, base, rows)


def column(name: str, ch, mart: str, seed: int, rows: np.ndarray,
           names_book=None) -> list:
    """Полное значение колонки, включая массивы и NULL."""
    base = key64(seed, mart, name)
    if is_embedding(name):
        n = len(rows)
        vals = np.round(normal(base, np.arange(n * EMBEDDING_DIM) + rows[0], 0, 0.4), 6)
        return [vals[i * EMBEDDING_DIM:(i + 1) * EMBEDDING_DIM].tolist()
                for i in range(n)]
    if getattr(ch, "is_array", False):
        lengths = integers(base ^ np.uint64(11), rows, 0, MAX_ARRAY)
        flat_rows = np.repeat(rows, lengths)
        flat = scalar(name, ch, base, flat_rows + np.arange(len(flat_rows)) * 7919,
                      names_book)
        # tolist(), а не list(): numpy-скаляры не сериализуются в JSON, и
        # ошибка всплыла бы не при сборке, а в ответе API.
        out, pos = [], 0
        for length in lengths:
            chunk = flat[pos:pos + length]
            out.append(chunk.tolist() if isinstance(chunk, np.ndarray) else list(chunk))
            pos += int(length)
        return out
    values = scalar(name, ch, base, rows, names_book)
    values = values.tolist() if isinstance(values, np.ndarray) else list(values)
    if getattr(ch, "nullable", False):
        mask = bernoulli(base ^ np.uint64(13), rows, NULL_RATE)
        values = [None if m else v for m, v in zip(mask, values)]
    return values
