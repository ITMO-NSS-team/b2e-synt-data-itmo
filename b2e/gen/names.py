"""Грамматика русского имени: род, склонение, отчество по правилу.

Почему это отдельный модуль, а не список строк
----------------------------------------------
В прежнем корпусе ``employee_last_name`` хранился в мужской форме, а
``employee_full_name`` — в женской, и фильтр ``employee_last_name = 'Балашова'``
не находил никого. Это не опечатка данных, а отсутствие морфологии: если
женская форма получается приписыванием «а» к готовой строке, то согласовать
колонки между собой негде.

Здесь фамилия — это **пара форм** (мужская, женская), а отчество выводится из
имени отца правилом. Тогда любая колонка, где нужна фамилия, берёт форму по полу
носителя, и рассогласование становится невозможным.
"""
from __future__ import annotations

import numpy as np

from . import dicts
from .rng import key64, pick, weighted, zipf_cum

#: Суффиксы, после которых корень считается уже готовой фамилией.
_COMPLETE = ("ов", "ев", "ин", "ый", "ий", "ко", "ых", "их", "ов", "ёв")


#: Мягкие и шипящие окончания основы: после них идёт «-ев», а не «-ов».
#: «ц» сюда не входит: Кузнецов, Скворцов, Стрельцов — все на «-ов».
_SOFT = ("ь", "й", "ч", "ш", "щ", "ж", "е", "и", "я")


def _suffix_fits(root: str, suffix: str) -> bool:
    """Сочетается ли суффикс с основой.

    Без этого правила выходят «Иванев» и «Кириллко»: комбинаторика даёт объём
    словаря, но морфология решает, какие сочетания вообще бывают. Для корпуса,
    где имена читает и цитирует агент, неправдоподобная фамилия — такой же
    дефект, как неправдоподобное число.
    """
    soft = root.endswith(_SOFT)
    if suffix == "ев":
        return soft
    if suffix == "ов":
        return not soft
    if suffix in ("ко", "енко"):
        return not soft and len(root) > 4
    return True


def _feminine(surname: str) -> str:
    """Женская форма фамилии."""
    if surname.endswith(("ский", "цкий", "ской")):
        return surname[:-2] + "ая"
    if surname.endswith(("ых", "их", "ко", "енко", "аго", "ово")):
        return surname          # несклоняемые
    if surname.endswith(("ов", "ёв", "ев", "ин", "ын")):
        return surname + "а"
    return surname + "а"


def build_surnames() -> list[tuple[str, str]]:
    """Собрать словарь фамилий как пары (мужская, женская).

    Порядок значим: он же порядок частот по Ципфу, поэтому первые корни —
    самые распространённые фамилии.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for root in dicts.SURNAME_ROOTS:
        forms = [root] if root.endswith(_COMPLETE) else [
            root + suffix for suffix, _ in dicts.SURNAME_SUFFIXES
            if _suffix_fits(root, suffix)]
        for masculine in forms:
            if masculine in seen:
                continue
            seen.add(masculine)
            out.append((masculine, _feminine(masculine)))
    return out


def patronymic(father_name: str, female: bool) -> str:
    """Отчество от имени отца.

    Правило покрывает подавляющее большинство имён; исключения перечислены в
    словаре, потому что «Илья → Ильич», а не «Ильевич».
    """
    irregular = dicts.PATRONYMIC_IRREGULAR.get(father_name)
    if irregular:
        return irregular[1] if female else irregular[0]
    if father_name.endswith("ий"):
        stem = father_name[:-2] + "ь"
        return stem + ("евна" if female else "евич")
    if father_name.endswith("й"):
        stem = father_name[:-1]
        return stem + ("евна" if female else "евич")
    if father_name.endswith("ь"):
        stem = father_name[:-1]
        return stem + ("евна" if female else "евич")
    if father_name.endswith(("а", "я")):
        stem = father_name[:-1]
        return stem + ("ична" if female else "ич")
    return father_name + ("овна" if female else "ович")


class NameBook:
    """Готовые массивы имён и векторный сбор ФИО по индексам.

    Собирается один раз на сборку корпуса: 300 000 обращений к морфологии стоят
    дороже самих данных, а вариантов всего несколько тысяч.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.surnames = build_surnames()
        self.male = list(dicts.MALE_NAMES)
        self.female = list(dicts.FEMALE_NAMES)
        # Отчество берётся от «имени отца» — того же мужского ряда, но с
        # собственным распределением: поколение отцов сдвинуто, поэтому список
        # намеренно тот же, а частоты другие.
        self.patronymic_m = [patronymic(n, False) for n in self.male]
        self.patronymic_f = [patronymic(n, True) for n in self.male]

        self._cum_surname = zipf_cum(len(self.surnames), s=1.02)
        self._cum_male = zipf_cum(len(self.male), s=0.85)
        self._cum_female = zipf_cum(len(self.female), s=0.85)
        self._cum_father = zipf_cum(len(self.male), s=0.70)

        self._sur_m = np.array([m for m, _ in self.surnames], dtype=object)
        self._sur_f = np.array([f for _, f in self.surnames], dtype=object)
        self._male = np.array(self.male, dtype=object)
        self._female = np.array(self.female, dtype=object)
        self._patr_m = np.array(self.patronymic_m, dtype=object)
        self._patr_f = np.array(self.patronymic_f, dtype=object)

    def draw(self, index: np.ndarray, is_female: np.ndarray) -> dict[str, np.ndarray]:
        """Породить ФИО для набора людей.

        Возвращает и собранное ФИО, и его части — именно потому, что части
        обязаны быть согласованы с целым: ``employee_last_name`` — это фамилия
        носителя в его роде, а не мужская форма.
        """
        s_idx = weighted(key64(self.seed, "surname"), index, self._cum_surname)
        f_idx = weighted(key64(self.seed, "father"), index, self._cum_father)
        m_idx = weighted(key64(self.seed, "given.m"), index, self._cum_male)
        w_idx = weighted(key64(self.seed, "given.f"), index, self._cum_female)

        last = np.where(is_female, self._sur_f[s_idx], self._sur_m[s_idx])
        first = np.where(is_female, self._female[w_idx], self._male[m_idx])
        middle = np.where(is_female, self._patr_f[f_idx], self._patr_m[f_idx])
        full = np.array([f"{a} {b} {c}" for a, b, c in zip(last, first, middle)],
                        dtype=object)
        return {"last": last, "first": first, "middle": middle, "full": full,
                "surname_index": s_idx}

    def short(self, index: np.ndarray, is_female: np.ndarray) -> np.ndarray:
        """Фамилия и инициалы: «Ковалёва А.П.» — формат кадровых списков."""
        parts = self.draw(index, is_female)
        return np.array([f"{l} {f[0]}.{m[0]}." for l, f, m in
                         zip(parts["last"], parts["first"], parts["middle"])],
                        dtype=object)


def random_names(seed: int, ns: str, n: int, female_rate: float = 0.5) -> dict:
    """Имена для сущностей, не входящих в штат: детей, преемников, кандидатов."""
    book = NameBook(seed)
    idx = np.arange(n)
    female = pick(key64(seed, ns, "sex"), idx, 100) < int(female_rate * 100)
    return book.draw(idx + int(key64(seed, ns) % np.uint64(10_000_000)), female)
