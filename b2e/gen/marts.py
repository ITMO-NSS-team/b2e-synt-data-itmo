"""Строковые пространства всех 37 витрин каталога.

Витрина описывается тем, **о чём её строка**: о человеке, о штатной позиции, о
подразделении, о назначении на обучение, о кандидате, о заявке на подбор или о
внешнем профиле. Всё остальное — колонки — приходит из цепочки разрешения.

Пустых витрин здесь нет. В прежнем корпусе 19 витрин из 37 были пустыми, и целые
классы вопросов оказывались без ответа не потому, что так задумано, а потому что
до них не дошли руки. Пустота как проверяемое свойство осталась, но она теперь
объявляется явно в ``EMPTY_BY_DESIGN`` — тогда «пустой ответ валиден» можно
отличить от «генератор не доделан».
"""
from __future__ import annotations

import numpy as np

from . import dicts
from .population import AS_OF_DAYS, Population, _days_to_iso
from .resolve import Frame, Resolver, person_uuid
from .rng import bernoulli, cells, integers, key64, normal, pick, unit, weighted

#: Витрины, которые по замыслу остаются пустыми: на них проверяется, умеет ли
#: агент сказать «данных нет» вместо того, чтобы их придумать.
EMPTY_BY_DESIGN = {"technical.memai_memmcp_config"}

#: Доля людей, попадающих в устаревшие ``_dep``-копии: копия неполна, и это
#: наблюдаемое свойство — маршрутизация в неё даёт тихо неполный ответ.
DEP_COVERAGE = 0.82

TOP_N = 600


def _sub(n: int, share: float) -> int:
    return max(1, int(n * share))


def _person_frame(key: str, p: Population, rows: np.ndarray | None = None,
                  extra: dict | None = None) -> Frame:
    rows = np.arange(p.n) if rows is None else rows
    return Frame(key=key, n=len(rows), person=rows, extra=extra or {},
                 unit=p["unit_id"][rows])


# ------------------------------------------------------------------ витрины

def employee_actual(key: str, p: Population, r: Resolver) -> Frame:
    return _person_frame(key, p)


def employee_dep(key: str, p: Population, r: Resolver) -> Frame:
    """Устаревшая копия: те же люди, но не все — копия отстала от источника."""
    keep = np.flatnonzero(bernoulli(key64(p.seed, "dep", key), np.arange(p.n),
                                    DEP_COVERAGE))
    return _person_frame(key, p, keep)


def employee_orion(key: str, p: Population, r: Resolver) -> Frame:
    """Канал orion: только сотрудники технологического блока."""
    tech = [i for i, (b, _) in enumerate(dicts.BLOCKS) if b == "Технологии"][0]
    keep = np.flatnonzero(p["block_idx"] == tech)
    return _person_frame(key, p, keep)


def employee_increment(key: str, p: Population, r: Resolver) -> Frame:
    """Инкремент: только те, у кого за последний месяц было изменение."""
    changed = ((p["position_start_days"] > AS_OF_DAYS - 31)
               | (p["hire_days"] > AS_OF_DAYS - 31)
               | (p["fired_days"] > AS_OF_DAYS - 31))
    return _person_frame(key, p, np.flatnonzero(changed))


def top600(key: str, p: Population, r: Resolver) -> Frame:
    """Топ-600: руководители высших грейдов. Отбор детерминирован."""
    order = np.lexsort((-p["engagement"], -p["grade_level"]))
    rows = order[:min(TOP_N, p.n)]
    comps = p["competencies"][rows]
    score_names = ["personality_score", "cognitive_score", "motivation_score",
                   "soft_skills_score", "management_score", "wide_context_score",
                   "influence_scale_score", "reflection_score",
                   "life_intelligence_score"]
    extra = {nm: comps[:, i] for i, nm in enumerate(score_names)}
    extra["professionlevel_name"] = np.array(
        ["Эксперт", "Лидер", "Профессионал"], dtype=object)[
            pick(key64(p.seed, "top.lvl"), np.arange(len(rows)), 3)]
    extra["labor_book_company_name"] = np.full(len(rows), "ПАО Сбербанк", dtype=object)
    extra["has_skills"] = np.ones(len(rows), dtype=np.int64)
    return _person_frame(key, p, rows, extra=extra)


def employee_education(key: str, p: Population, r: Resolver) -> Frame:
    """Развёртка назначений на обучение: строка — одно назначение."""
    block = r.blocks["learning"]
    rows = block.flat_rows()
    flat = np.arange(block.total)
    extra = {
        "assignment_id": np.array([str(v) for v in block.fields["assignment_id"]],
                                  dtype=object),
        "assignment_status": block.fields["assignment_status"],
        "item_name": block.fields["item_name"],
        "item_type": block.fields["item_type"],
        "item_format_name": block.fields["item_format_name"],
        "item_duration": block.fields["item_duration"],
        "item_assigned_dt": _days_to_iso(block.fields["assigned_days"]),
        "item_deadline_dt": _days_to_iso(block.fields["deadline_days"]),
        "item_completed_dt": np.array(
            [None if d < 0 else _days_to_iso(np.array([d]))[0]
             for d in block.fields["completed_days"]], dtype=object),
        "playlist_name": np.array([f"Трек «{n}»" for n in block.fields["item_name"]],
                                  dtype=object),
    }
    return Frame(key=key, n=block.total, person=rows, extra=extra,
                 unit=p["unit_id"][rows])


def employee_oshs(key: str, p: Population, r: Resolver) -> Frame:
    return _person_frame(key, p)


def position_frame(key: str, p: Population, r: Resolver) -> Frame:
    """Штатные позиции: занятые сотрудниками плюс вакантные."""
    from .org import VACANCY_RATE
    extra_n = int(p.n * VACANCY_RATE)
    person = np.r_[np.arange(p.n), np.full(extra_n, -1)]
    occupied = person >= 0
    units_free = p.tree.leaves[pick(key64(p.seed, "vac.unit"), np.arange(extra_n),
                                    len(p.tree.leaves))]
    units = np.r_[p["unit_id"], units_free]
    ids = np.r_[p["position_id"], 900_000 + np.arange(extra_n)]
    extra = {
        "position_id": np.array([str(int(v)) for v in ids], dtype=object),
        "vacancy": np.where(occupied, 0.0,
                            np.round(0.5 + 0.5 * unit(key64(p.seed, "vac.rate"),
                                                      np.arange(len(ids))), 2)),
        "position_free": (~occupied).astype(np.float64),
        "vacancy_clear_flag": (~occupied).astype(np.int64),
        "occupied_flag": occupied.astype(np.int64),
    }
    return Frame(key=key, n=len(ids), person=person, extra=extra, unit=units)


def org_structure(key: str, p: Population, r: Resolver) -> Frame:
    """Подразделения: строка — узел дерева."""
    tree = p.tree
    n = len(tree)
    heads = tree.head_person
    extra = {
        "oshs_unit_id": np.array([str(1_000_000 + i) for i in range(n)], dtype=object),
        "oshs_unit_name": tree.name,
        "unit_name": tree.name,
        "oshs_unit_level": tree.level,
        "oshs_parent_unit_id": np.array(
            [None if v < 0 else str(1_000_000 + int(v)) for v in tree.parent],
            dtype=object),
        "oshs_unit_type": tree.kind,
        "block_name": np.array([b[0] for b in dicts.BLOCKS], dtype=object)[tree.block],
        "city_name": np.array([c[0] for c in dicts.CITIES], dtype=object)[tree.city],
        "oshs_headcount": tree.headcount,
        "oshs_hrbp_employee_id": np.array(
            [None if h < 0 else str(int(p["employee_id"][h])) for h in heads],
            dtype=object),
    }
    return Frame(key=key, n=n, person=np.where(heads >= 0, heads, -1), extra=extra,
                 unit=np.arange(n))


def competence(key: str, p: Population, r: Resolver) -> Frame:
    """Компетенции: девять различимых оценок вместо девяти одинаковых."""
    comps = p["competencies"]
    names = ["personality_traits_group_score", "cognitive_features_group_score",
             "soft_skills_competency_score", "management_competency_score",
             "wide_context", "influence_scale", "reflection", "life_intelligenc",
             "extrinsic_motivation"]
    extra = {nm: comps[:, i] for i, nm in enumerate(names)}
    extra["competency_avg"] = comps.mean(axis=1).round(3)
    return _person_frame(key, p, extra=extra)


def churn(key: str, p: Population, r: Resolver) -> Frame:
    """Признаки модели оттока плюс сам риск — прежде он нигде не сохранялся."""
    idx = np.arange(p.n)
    absence_days = r.blocks["absence"].counts * 4
    extra = {
        "attrition_risk": np.array(["Низкий", "Средний", "Высокий"], dtype=object)[
            p["attrition_risk_idx"]],
        "churn_probability": np.round(
            0.04 + 0.16 * p["attrition_risk_idx"]
            + 0.03 * normal(key64(p.seed, "churn.p"), idx), 4).clip(0.001, 0.98),
        "illness_days_12m_sum": absence_days,
        "days_since_last_vacation": integers(key64(p.seed, "churn.vac"), idx, 10, 400),
        "vacation_next_month_flag": bernoulli(key64(p.seed, "churn.next"), idx,
                                              0.12).astype(np.int64),
        "pos_grade_num": p["grade_level"],
        "child_count": p["children_qty"],
    }
    return _person_frame(key, p, extra=extra)


def talent_radar(key: str, p: Population, r: Resolver) -> Frame:
    """Внешние профили: часть — зеркала сотрудников, часть — люди со стороны.

    Именно эта витрина делает исследуемым вопрос о переносе памяти между
    системами: один человек присутствует и здесь, и в кадровой витрине, но под
    другим ключом и с частично расходящимися атрибутами.
    """
    mirrored = _sub(p.n, 0.10)
    external = _sub(p.n, 0.25)
    order = np.lexsort((-p["star_index"], -p["grade_level"]))[:mirrored]
    person = np.r_[order, np.full(external, -1)]
    n = len(person)
    idx = np.arange(n)
    ext_names = p.book.draw(idx + 5_000_000,
                            bernoulli(key64(p.seed, "tr.sex"), idx, 0.45))
    star_mirror = p["star_index"][order]
    star_ext = np.where(bernoulli(key64(p.seed, "tr.scored"), np.arange(external), 0.8),
                        integers(key64(p.seed, "tr.star"), np.arange(external), 5, 100),
                        -1)
    star = np.r_[star_mirror, star_ext]
    has_emb = bernoulli(key64(p.seed, "tr.emb"), idx, 0.74)
    extra = {
        "id": person_uuid(p, idx + 9_000_000),
        "full_name": np.where(person >= 0, p["full_name"][np.maximum(person, 0)],
                              ext_names["full"]),
        "star_index": [None if v < 0 else int(v) for v in star],
        "score_total": [None if v < 0 else int(v) for v in star],
        "has_embedding": has_emb,
        "region": np.array(dicts.REGION_CODES, dtype=object)[
            weighted(key64(p.seed, "tr.reg"), idx,
                     np.cumsum(np.array([0.55, 0.12, 0.11, 0.07, 0.07, 0.05, 0.03])))],
        "primary_domain": np.array(sorted(dicts.JOB_FAMILIES), dtype=object)[
            pick(key64(p.seed, "tr.dom"), idx, len(dicts.JOB_FAMILIES))],
        "current_company": np.where(
            person >= 0, "ПАО Сбербанк",
            np.array(dicts.EXT_COMPANIES, dtype=object)[
                pick(key64(p.seed, "tr.co"), idx, len(dicts.EXT_COMPANIES))]),
        "is_employee": (person >= 0).astype(np.int64),
    }
    return Frame(key=key, n=n, person=person, extra=extra)


def candidates(key: str, p: Population, r: Resolver) -> Frame:
    """Кандидаты: внешние люди плюс внутренние переходы."""
    n = _sub(p.n, 0.09)
    idx = np.arange(n)
    internal = bernoulli(key64(p.seed, "cand.int"), idx, 0.28)
    person = np.where(internal, pick(key64(p.seed, "cand.who"), idx, p.n), -1)
    names = p.book.draw(idx + 3_000_000, bernoulli(key64(p.seed, "cand.sex"), idx, 0.55))
    stages = ["Новый", "Скрининг", "Интервью с HR", "Техническое интервью",
              "Финальное интервью", "Оффер", "Принят", "Отказ"]
    extra = {
        "candidate_id": person_uuid(p, idx + 7_000_000),
        "full_name": np.where(person >= 0, p["full_name"][np.maximum(person, 0)],
                              names["full"]),
        "first_name": np.where(person >= 0, p["first_name"][np.maximum(person, 0)],
                               names["first"]),
        "last_name": np.where(person >= 0, p["last_name"][np.maximum(person, 0)],
                              names["last"]),
        "stage": np.array(stages, dtype=object)[
            weighted(key64(p.seed, "cand.stage"), idx, np.cumsum(
                np.array([0.18, 0.16, 0.14, 0.12, 0.10, 0.07, 0.08, 0.15])))],
        "is_internal": internal.astype(np.int64),
        "source": np.array(["hh.ru", "Рекомендация сотрудника", "Карьерный сайт",
                            "Хантинг", "Внутренний портал", "Стажировка"],
                           dtype=object)[
            pick(key64(p.seed, "cand.src"), idx, 6)],
    }
    return Frame(key=key, n=n, person=person, extra=extra)


def requisitions(key: str, p: Population, r: Resolver) -> Frame:
    """Заявки на подбор — по одной на вакантную позицию."""
    from .org import VACANCY_RATE
    n = _sub(p.n, VACANCY_RATE)
    idx = np.arange(n)
    units = p.tree.leaves[pick(key64(p.seed, "req.unit"), idx, len(p.tree.leaves))]
    fam = np.array(sorted(dicts.JOB_FAMILIES), dtype=object)[
        pick(key64(p.seed, "req.fam"), idx, len(dicts.JOB_FAMILIES))]
    roles = np.array([dicts.JOB_FAMILIES[f][
        int(pick(key64(p.seed, "req.role"), np.array([i]), len(dicts.JOB_FAMILIES[f]))[0])]
        for i, f in zip(idx, fam)], dtype=object)
    opened = AS_OF_DAYS - integers(key64(p.seed, "req.open"), idx, 3, 240)
    extra = {
        "requisition_id": np.array([f"REQ-{2026}-{100000 + i}" for i in idx],
                                   dtype=object),
        "position_name": roles,
        "job_family": fam,
        "unit_name": p.tree.name[units],
        "city_name": np.array([c[0] for c in dicts.CITIES], dtype=object)[
            p.tree.city[units]],
        "grade_min": (8 + pick(key64(p.seed, "req.gmin"), idx, 6)).astype(np.int64),
        "opened_date": _days_to_iso(opened),
        "days_open": (AS_OF_DAYS - opened).astype(np.int64),
        "status": np.array(["Открыта", "На согласовании", "В работе",
                            "Закрыта", "Приостановлена"], dtype=object)[
            weighted(key64(p.seed, "req.st"), idx,
                     np.cumsum(np.array([0.34, 0.10, 0.30, 0.20, 0.06])))],
        "priority": np.array(["Высокий", "Средний", "Низкий"], dtype=object)[
            weighted(key64(p.seed, "req.pr"), idx,
                     np.cumsum(np.array([0.22, 0.55, 0.23])))],
    }
    return Frame(key=key, n=n, person=np.full(n, -1), extra=extra, unit=units)


def publications(key: str, p: Population, r: Resolver) -> Frame:
    """Публикации вакансий: производная от заявок."""
    from .org import VACANCY_RATE
    n = _sub(p.n, VACANCY_RATE * 0.8)
    idx = np.arange(n)
    fam = np.array(sorted(dicts.JOB_FAMILIES), dtype=object)[
        pick(key64(p.seed, "pub.fam"), idx, len(dicts.JOB_FAMILIES))]
    extra = {
        "publication_id": np.array([f"PUB-{200000 + i}" for i in idx], dtype=object),
        "requisition_id": np.array([f"REQ-2026-{100000 + int(i)}" for i in
                                    pick(key64(p.seed, "pub.req"), idx, max(n, 1))],
                                   dtype=object),
        "title": np.array([dicts.JOB_FAMILIES[f][0] for f in fam], dtype=object),
        "channel": np.array(["hh.ru", "Карьерный сайт", "Telegram", "VK Работа",
                             "Хабр Карьера"], dtype=object)[
            pick(key64(p.seed, "pub.ch"), idx, 5)],
        "views": integers(key64(p.seed, "pub.v"), idx, 30, 12000),
        "responses": integers(key64(p.seed, "pub.r"), idx, 0, 400),
        "content": np.array(
            [f"Приглашаем в команду на позицию «{dicts.JOB_FAMILIES[f][0]}». "
             f"Требования: профильное образование, опыт от двух лет, "
             f"готовность работать в кросс-функциональной команде."
             for f in fam], dtype=object),
        "extracted_attributes": np.array(
            [{"family": str(f), "remote": int(i % 3 == 0)} for i, f in
             zip(idx, fam)], dtype=object),
    }
    return Frame(key=key, n=n, person=np.full(n, -1), extra=extra)


def evolution(key: str, p: Population, r: Resolver) -> Frame:
    """Каталог элементов развития: строка — курс или программа, а не человек.

    ``reaction.person_id`` — реакции сотрудников на элемент; это реальные люди,
    поэтому «что рекомендуют людям вроде меня» становится отвечаемым вопросом.
    """
    items = [c[0] for c in dicts.COURSES]
    n = len(items)
    idx = np.arange(n)
    duration = np.array([c[2] for c in dicts.COURSES])

    react = []
    for i in idx:
        k = int(integers(key64(p.seed, "evo.rn"), np.array([i]), 0, 12)[0])
        who = pick(key64(p.seed, "evo.who"), np.arange(k) + i * 100, p.n)
        react.append(person_uuid(p, who).tolist())
    extra = {
        "item_id": np.array([f"ITEM-{3000 + i}" for i in idx], dtype=object),
        "name": np.array(items, dtype=object),
        "short_description": np.array([f"Программа развития: {x.lower()}"
                                       for x in items], dtype=object),
        "description": np.array(
            [f"Курс «{x}» рассчитан на {d} академических часов и завершается "
             f"проверкой знаний." for x, d in zip(items, duration)], dtype=object),
        "type": np.array([c[1] for c in dicts.COURSES], dtype=object),
        "duration": duration,
        "status": np.array(["Опубликован", "Черновик", "Архив"], dtype=object)[
            weighted(key64(p.seed, "evo.st"), idx,
                     np.cumsum(np.array([0.82, 0.1, 0.08])))],
        "category": np.array(["Обязательное", "Профессиональное", "Управленческое",
                              "Цифровые навыки"], dtype=object)[
            pick(key64(p.seed, "evo.cat"), idx, 4)],
        "source_system": np.full(n, "Виртуальная школа", dtype=object),
        "score": np.round(3.5 + 1.4 * unit(key64(p.seed, "evo.s"), idx), 2),
        "reaction.person_id": np.array(react, dtype=object),
        "company": np.full(n, "paosberbank", dtype=object),
    }
    return Frame(key=key, n=n, person=np.full(n, -1), extra=extra)


def digital_trail(key: str, p: Population, r: Resolver) -> Frame:
    """Цифровой след компетенций: активность в корпоративных системах."""
    n = _sub(p.n, 0.55)
    idx = np.arange(n)
    person = pick(key64(p.seed, "trail.who"), idx, p.n)
    extra = {
        "competence_name": np.array(dicts.COMPETENCIES, dtype=object)[
            pick(key64(p.seed, "trail.c"), idx, len(dicts.COMPETENCIES))],
        "events_count": integers(key64(p.seed, "trail.e"), idx, 1, 240),
        "source_system": np.array(["Пульс", "Jira", "Confluence", "Битрикс",
                                   "Виртуальная школа"], dtype=object)[
            pick(key64(p.seed, "trail.s"), idx, 5)],
    }
    return Frame(key=key, n=n, person=person, extra=extra)


def agentic_skill(key: str, p: Population, r: Resolver) -> Frame:
    """Уровень владения ИИ-инструментами — прямо релевантно B2E-эксперименту."""
    n = _sub(p.n, 0.6)
    idx = np.arange(n)
    person = pick(key64(p.seed, "ai.who"), idx, p.n)
    level = weighted(key64(p.seed, "ai.lvl"), idx,
                     np.cumsum(np.array([0.30, 0.34, 0.22, 0.10, 0.04])))
    # Шесть ступеней владения ИИ-инструментами: от промптинга до работы с
    # командой агентов. Ступень не даётся через одну — уровень монотонен.
    steps = ["n1_prompting", "n2_prompt_engineering", "n3_context_managment",
             "n4_creating_tools", "n5_creating_skills", "n6_agent_team"]
    reached = np.clip(level + 1, 0, len(steps))
    extra = {
        "tn": np.array([str(int(v)) for v in p["employee_id"][person]], dtype=object),
        "calc_date": np.full(n, _days_to_iso(np.array([AS_OF_DAYS]))[0], dtype=object),
        "expertise_area": np.array(sorted(dicts.JOB_FAMILIES), dtype=object)[
            p["family_idx"][person]],
        "role_main": np.array(["Разработчик", "Аналитик", "Руководитель",
                               "Специалист поддержки", "Продуктовая роль"],
                              dtype=object)[pick(key64(p.seed, "ai.role"), idx, 5)],
        "use_mode": np.array(["Не использует", "Эпизодически", "В рабочем потоке",
                              "Ежедневно"], dtype=object)[np.clip(level, 0, 3)],
        "use_frequency": (level * 6 + integers(key64(p.seed, "ai.s"), idx, 0, 9)),
    }
    for i, step in enumerate(steps):
        extra[step] = (reached > i).astype(np.int64)
    return Frame(key=key, n=n, person=person, extra=extra)


def embeddings(key: str, p: Population, r: Resolver) -> Frame:
    """Векторные индексы: по оргструктуре — узлы дерева, по навыкам — навыки."""
    if "skills" in key:
        pool = sorted({s for v in dicts.SKILLS_BY_FAMILY.values() for s in v})
        n = len(pool)
        return Frame(key=key, n=n, person=np.full(n, -1), extra={
            "structure_name": np.array(pool, dtype=object),
            "skill_name": np.array(pool, dtype=object),
        })
    nodes = p.tree.unit_id
    n = len(nodes)
    return Frame(key=key, n=n, person=np.full(n, -1), unit=nodes, extra={
        "structure_name": p.tree.name,
        "structure_type": p.tree.kind,
        "structure_code": np.array([str(1_000_000 + int(v)) for v in nodes],
                                   dtype=object),
    })


def checkins(key: str, p: Population, r: Resolver) -> Frame:
    """Структура расшифровок встреч один на один."""
    n = _sub(p.n, 0.18)
    idx = np.arange(n)
    person = pick(key64(p.seed, "chk.who"), idx, p.n)
    topics = np.array(["Цели на квартал", "Обратная связь", "Развитие",
                       "Карьерный трек", "Нагрузка и приоритеты", "Итоги проекта"],
                      dtype=object)[pick(key64(p.seed, "chk.t"), idx, 6)]
    names = p["full_name"][person]
    extra = {
        "id": person_uuid(p, idx + 11_000_000),
        "meeting_id": np.array([f"MTG-{700000 + i}" for i in idx], dtype=object),
        "event_type": np.full(n, "checkin", dtype=object),
        "created_at": np.array(
            [f"{d} 10:00:00" for d in _days_to_iso(
                AS_OF_DAYS - integers(key64(p.seed, "chk.d"), idx, 1, 400))],
            dtype=object),
        "status": np.array(["processed", "processed", "failed"], dtype=object)[
            weighted(key64(p.seed, "chk.s"), idx, np.cumsum(np.array([0.9, 0.07, 0.03])))],
        "preprocessed_text": np.array(
            [f"Встреча один на один с {nm}. Обсудили: {t.lower()}."
             for nm, t in zip(names, topics)], dtype=object),
        "structure_text": np.array(
            [f"Тема: {t}. Договорённости зафиксированы, срок — следующий квартал."
             for t in topics], dtype=object),
        "agenda_items": np.array([[t, "Обратная связь", "Планы"] for t in topics],
                                 dtype=object),
        "manual_agreements": np.array(
            [[["Подготовить план развития"], ["Уточнить цели по кварталу"], []][int(i) % 3]
             for i in idx], dtype=object),
    }
    return Frame(key=key, n=n, person=person, extra=extra)


def cladr(key: str, p: Population, r: Resolver) -> Frame:
    """Классификатор адресов: справочник городов присутствия."""
    n = len(dicts.CITIES)
    idx = np.arange(n)
    names = np.array([c[0] for c in dicts.CITIES], dtype=object)
    extra = {
        "code": np.array([f"{i + 1:02d}000000000000" for i in idx], dtype=object),
        "city_name": names,
        "position_city": names,
        "area_name": np.array([f"{c} городской округ" for c in names], dtype=object),
        "region_name": np.array([f"{c} и область" for c in names], dtype=object),
        "city_ordering": idx + 1,
    }
    return Frame(key=key, n=n, person=np.full(n, -1), extra=extra)


def memai_qa(key: str, p: Population, r: Resolver) -> Frame:
    """Вопросы и ответы корпоративной базы знаний."""
    questions = [
        "Как оформить отпуск?", "Когда индексируется зарплата?",
        "Как записаться на обучение?", "Что делать при переводе в другой город?",
        "Как получить справку 2-НДФЛ?", "Какой порядок оценки по целям?",
        "Как попасть в кадровый резерв?", "Кто мой HR бизнес-партнёр?",
        "Как оформить больничный?", "Что входит в ДМС?",
        "Как подать заявку на подбор?", "Какие льготы у сотрудников с детьми?",
    ]
    answers = [
        "Заявка оформляется в кадровом портале, согласование — у руководителя.",
        "Пересмотр оплаты проходит ежегодно по итогам оценки результативности.",
        "Каталог обучения доступен в Виртуальной школе, назначение — через руководителя.",
        "Перевод оформляется через HR бизнес-партнёра принимающего подразделения.",
        "Справка заказывается в кадровом портале, срок подготовки — три рабочих дня.",
        "Цели ставятся на квартал, оценка — по шкале A–E по результату и ценностям.",
        "Отбор проходит ежегодно по представлению руководителя и итогам оценки.",
        "HR бизнес-партнёр закреплён за подразделением, контакт — в карточке ОШС.",
        "Электронный лист нетрудоспособности поступает автоматически.",
        "Программа ДМС включает поликлинику, стоматологию и телемедицину.",
        "Заявка создаётся в системе подбора, согласуется руководителем и HR.",
        "Дополнительные дни отпуска и материальная помощь при рождении ребёнка.",
    ]
    # Витрина по схеме каталога — это диалог скрининга кандидата ИИ-агентом,
    # а не справочник. Вопросы берутся из корпоративной базы знаний, но
    # укладываются в поля скрининга: вопрос, ответ, оценка, объяснение.
    reps = max(1, _sub(p.n, 0.02) // len(questions))
    idx = np.arange(len(questions) * reps)
    n = len(idx)
    q = np.array(questions * reps, dtype=object)
    a = np.array(answers * reps, dtype=object)
    score = np.round(0.35 + 0.6 * unit(key64(p.seed, "qa.score"), idx), 3)
    return Frame(key=key, n=n, person=np.full(n, -1), extra={
        "dialog_question": q,
        "dialog_answer": a,
        "dialog_order": (idx % len(questions)) + 1,
        "question_type": np.array(["Уточняющий", "Проверочный", "Открытый"],
                                  dtype=object)[pick(key64(p.seed, "qa.t"), idx, 3)],
        "score": score,
        "explanation": np.where(score > 0.7, "Ответ полный, источник указан.",
                                "Ответ неполный, требуется уточнение."),
        "summary": np.array([f"Диалог по теме «{x[:28]}…»" for x in q], dtype=object),
        "agent_name": np.full(n, "b2e-assistant", dtype=object),
        "agent_version": np.full(n, "1.4.0", dtype=object),
        "model_name": np.full(n, "haiku-4.5", dtype=object),
    })


def position_dict(key: str, p: Population, r: Resolver) -> Frame:
    """Справочник должностей: одна строка на уникальную роль."""
    roles = sorted({role for family in dicts.JOB_FAMILIES.values() for role in family})
    n = len(roles)
    idx = np.arange(n)
    return Frame(key=key, n=n, person=np.full(n, -1), extra={
        "position_id": np.array([str(500_000 + i) for i in idx], dtype=object),
        "position_name": np.array(roles, dtype=object),
        "grade_min": (8 + pick(key64(p.seed, "dict.gmin"), idx, 5)).astype(np.int64),
        "grade_max": (13 + pick(key64(p.seed, "dict.gmax"), idx, 6)).astype(np.int64),
    })


#: Витрина → построитель строкового пространства.
BUILDERS = {
    "dm_core.employee_actual": employee_actual,
    "dm_core.employee": employee_actual,
    "dm_core.employee_hist": employee_actual,
    "dm_core.employee_actual_increment": employee_increment,
    "dm_core.employee_actual_orion": employee_orion,
    "dm_core.employee_education": employee_education,
    "dm_core.employee_oshs": employee_oshs,
    "dm_core.position_actual": position_frame,
    "dm_core.position_hist": position_frame,
    "dm_core.candidate_actual": candidates,
    "dm_core.candidate_hist": candidates,
    "dm_core.recruitment_actual": candidates,
    "dm_core.org_structure": org_structure,
    "dm_core.evolution_item": evolution,
    "dm_special.employee_competence_actual": competence,
    "dm_special.competence_digital_trail": digital_trail,
    "dm_special.talent_radar_people": talent_radar,
    "dm_special.churn_model_metrics": churn,
    "dm_special.top_600_stats": top600,
    "dm_special.staff_employee_digital_profile_top_600": top600,
    "dm_special.memai_qa": memai_qa,
    "recruitment.job_requisition_large": requisitions,
    "recruitment.publication": publications,
    "recruitment.publication_extract": publications,
    "recruitment.publication_extract_enriched": publications,
    "anagent.employee_actual_dep": employee_dep,
    "anagent.employee_hist_dep": employee_dep,
    "anagent.employee_actual_orion_dep": employee_orion,
    "anagent.recruitment_sint": candidates,
    "anagent.oss_ebase_embs": embeddings,
    "anagent.oss_ebase_embs_dep": embeddings,
    "anagent.skills_ebase_embs_dep": embeddings,
    "sset.staff_position_dict": position_dict,
    "stable.agentic_skill_level": agentic_skill,
    "stable.checkins_transcribation_structure": checkins,
    "stable.cladr_code_actual": cladr,
    "technical.memai_memmcp_config": None,        # пусто по замыслу
}


def frame_for(key: str, p: Population, r: Resolver) -> Frame | None:
    builder = BUILDERS.get(key)
    if builder is None:
        return None
    return builder(key, p, r)
