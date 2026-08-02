"""Реестр каверз: тихие ошибки, которые данные допускают намеренно.

Зачем это отдельный переключаемый слой
--------------------------------------
Каверза — это ситуация, где запрос возвращает 200 OK и неверный ответ.
Их наличие делает симулятор пригодным для измерения галлюцинаций. Но если
каверзы вшиты в генератор намертво, нельзя ответить на главный вопрос
исследования: **насколько именно они виноваты**. Поэтому каждая объявляется
флагом, и корпус собирается дважды — с ними и без.

Каверзы применяются к уже готовым значениям колонки, а не к модели предметной
области: они искажают представление, а не факт.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from b2e.gen.rng import bernoulli, key64


@dataclass(frozen=True)
class Trap:
    name: str
    column: str
    description: str
    enabled: bool = True


#: Реестр. Расширяется по мере находок в боевых трейсах агента.
REGISTRY: tuple[Trap, ...] = (
    Trap("upper_cyrillic", "employee_full_name",
         "Часть ФИО записана заглавными. ClickHouse ILIKE не сворачивает "
         "кириллицу, поэтому фильтр «%петров%» их молча не найдёт."),
    Trap("null_means_unscored", "star_index",
         "NULL значит «не оценивался», а не «слабый». Слияние с низкими "
         "значениями даёт неверный вывод без всякой ошибки."),
    Trap("duplicate_signal", "score_total",
         "score_total — дубль star_index, а не второй независимый сигнал. "
         "Подача их как двух подтверждений завышает уверенность."),
    Trap("raw_region_codes", "region",
         "Регион хранится сырыми кодами (am/eu/us). Фильтр region = 'eu' "
         "молча теряет профили, записанные соседними кодами."),
    Trap("stale_key_employee", "is_key_employee",
         "Флаг ключевого сотрудника — прошлогодний срез методики и намеренно "
         "не равен расчёту по текущим данным."),
)

DEFAULT_ENABLED = {"upper_cyrillic", "null_means_unscored", "duplicate_signal",
                   "raw_region_codes", "stale_key_employee"}

#: Доля записей, затронутых регистровой каверзой.
UPPER_RATE = 0.11


def apply_upper(values: list, seed: int, person: np.ndarray, enabled: bool) -> list:
    """Перевести часть ФИО в верхний регистр.

    Ключ каверзы — **человек**, а не строка витрины. Иначе один и тот же
    сотрудник оказался бы в верхнем регистре на одной витрине и в нижнем на
    другой, и каверза превратилась бы в рассогласование личности — то есть в
    дефект W1, который проект как раз устраняет. Каверза обязана быть
    воспроизводимым свойством записи, а не шумом представления.
    """
    if not enabled:
        return values
    mask = bernoulli(key64(seed, "trap.upper"), np.maximum(person, 0), UPPER_RATE)
    mask = mask & (person >= 0)
    return [v.upper() if (m and isinstance(v, str)) else v
            for m, v in zip(mask, values)]


def apply_upper_nested(values: list, seed: int, refs: list, n_people: int,
                       enabled: bool) -> list:
    """Та же каверза для ФИО внутри вложенного массива.

    Ключ — человек, НА КОТОРОГО ссылается элемент, а не строка витрины. Иначе
    один и тот же сотрудник был бы «ИВАНОВ» в ``employee_full_name`` и «Иванов»
    в чужом ``successors.full_name``: как только имена преемников стали
    настоящими, забыть про каверзу здесь означало бы вернуть дефект W1 —
    рассогласование личности — через заднюю дверь.
    """
    if not enabled:
        return values
    mask = bernoulli(key64(seed, "trap.upper"), np.arange(n_people), UPPER_RATE)
    out = []
    for item, ids in zip(values, refs):
        if not item:
            out.append(item)
            continue
        out.append([v.upper() if (mask[i] and isinstance(v, str)) else v
                    for i, v in zip(ids, item)])
    return out


def describe(enabled: set[str] | None = None) -> str:
    """Человекочитаемый реестр — идёт в документацию корпуса."""
    on = DEFAULT_ENABLED if enabled is None else enabled
    lines = ["| каверза | колонка | включена | что ломает |",
             "|---|---|:--:|---|"]
    for trap in REGISTRY:
        lines.append(f"| `{trap.name}` | `{trap.column}` | "
                     f"{'да' if trap.name in on else 'нет'} | {trap.description} |")
    return "\n".join(lines)
