"""Организационное дерево: блок → территориальный банк → департамент → … → команда.

Почему дерево строится первым
-----------------------------
В прежнем корпусе руководители, преемники и «дети» были придуманными строками:
``successors.person_id`` не указывал ни на одну существующую строку, а
``org_structure`` жила отдельно от сотрудников. Из-за этого анализ команды —
основной сценарий B2E-агента — был невыполним в принципе.

Здесь порядок обратный: сначала строится дерево подразделений и штатные
позиции, потом люди занимают позиции, и только потом руководителем
подразделения назначается **реальный сотрудник из него**. Ссылочная целостность
получается конструктивно, а не проверкой постфактум.

Глубина. Каталог Heimdall несёт уровни ``oshs_level_1..15``. Реальные ветки
неоднородны: розница глубже технологий. Дерево здесь пятиуровневое, уровни
6–15 пустые — это честнее, чем заполнять их шумом.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dicts
from .rng import integers, key64, pick, unit, weighted

#: Средний размер команды (лист дерева) и подразделений выше по уровню.
LEAF_SIZE = 12
DIVISION_SIZE = 60
DEPARTMENT_SIZE = 320

#: Доля вакантных позиций сверх занятых.
VACANCY_RATE = 0.075


@dataclass
class OrgTree:
    """Плоское представление дерева: массивы одинаковой длины, индекс = unit_id."""

    unit_id: np.ndarray
    name: np.ndarray
    kind: np.ndarray
    level: np.ndarray
    parent: np.ndarray            # индекс родителя, -1 у корня
    block: np.ndarray             # индекс блока
    tb: np.ndarray                # индекс территориального банка
    city: np.ndarray              # индекс города
    headcount: np.ndarray         # плановая численность (только у листьев > 0)
    leaves: np.ndarray            # индексы листовых подразделений
    head_person: np.ndarray = field(default=None)  # заполняется после найма

    def __len__(self) -> int:
        return len(self.unit_id)

    def path(self, node: int) -> list[int]:
        """Путь от корня до узла — источник для ``oshs_level_N_unit_name``."""
        out: list[int] = []
        while node >= 0:
            out.append(node)
            node = int(self.parent[node])
        return list(reversed(out))

    def ancestors_matrix(self, max_level: int = 15) -> np.ndarray:
        """Матрица «узел × уровень» → unit_id предка на этом уровне (или -1).

        Считается один раз: 300 000 обращений к ``path`` на каждую из
        пятнадцати колонок уровня — это 4,5 млн проходов по дереву.
        """
        n = len(self)
        out = np.full((n, max_level), -1, dtype=np.int64)
        order = np.argsort(self.level, kind="stable")
        for node in order:
            parent = int(self.parent[node])
            lvl = int(self.level[node]) - 1
            if parent >= 0:
                out[node] = out[parent]
            if lvl < max_level:
                out[node, lvl] = node
        return out


def _weights(pairs: list[tuple[str, float]]) -> np.ndarray:
    w = np.array([p[1] for p in pairs], dtype=np.float64)
    return np.cumsum(w / w.sum())


def build(seed: int, n_people: int) -> OrgTree:
    """Построить дерево под заданную численность."""
    blocks = [b for b, _ in dicts.BLOCKS]
    tbs = [t for t, _ in dicts.TERRITORIAL_BANKS]
    cities = [c for c, _ in dicts.CITIES]
    city_cum = _weights(dicts.CITIES)

    name: list[str] = []
    kind: list[str] = []
    level: list[int] = []
    parent: list[int] = []
    block: list[int] = []
    tb: list[int] = []
    city: list[int] = []
    headcount: list[int] = []

    def add(nm: str, kd: str, lv: int, pa: int, bl: int, tbi: int, ct: int,
            hc: int = 0) -> int:
        name.append(nm); kind.append(kd); level.append(lv); parent.append(pa)
        block.append(bl); tb.append(tbi); city.append(ct); headcount.append(hc)
        return len(name) - 1

    # Уровень 1 — блоки.
    root_of_block = {}
    for bi, b in enumerate(blocks):
        ci = int(weighted(key64(seed, "org.city.block"), np.array([bi]), city_cum)[0])
        root_of_block[bi] = add(b, "Блок", 1, -1, bi, -1, ci)

    counter = 0

    def split(total: int, typical: int, cap: int = 4096) -> list[int]:
        """Разбить численность на части около ``typical`` с разбросом."""
        nonlocal counter
        parts: list[int] = []
        left = total
        while left > 0 and len(parts) < cap:
            counter += 1
            jitter = float(unit(key64(seed, "org.split"), np.array([counter]))[0])
            size = max(3, int(typical * (0.55 + 0.9 * jitter)))
            size = min(size, left)
            if left - size < typical * 0.4:
                size = left
            parts.append(size)
            left -= size
        if left > 0:
            parts[-1] += left
        return parts

    # Уровни 2–5.
    for bi, (b, bw) in enumerate(dicts.BLOCKS):
        block_head = int(round(n_people * bw))
        for ti, (t, tw) in enumerate(dicts.TERRITORIAL_BANKS):
            share = int(round(block_head * tw))
            if share < 8:
                continue
            counter += 1
            ci = int(weighted(key64(seed, "org.city.tb"),
                              np.array([ti * 97 + bi]), city_cum)[0])
            tb_node = add(f"{b} — {t}", "Территориальный банк", 2,
                          root_of_block[bi], bi, ti, ci)
            for dep_size in split(share, DEPARTMENT_SIZE):
                counter += 1
                topic = dicts.UNIT_TOPICS[int(pick(key64(seed, "org.topic"),
                                                   np.array([counter]),
                                                   len(dicts.UNIT_TOPICS))[0])]
                dep = add(f"Департамент {topic}", "Департамент", 3, tb_node,
                          bi, ti, ci)
                for div_size in split(dep_size, DIVISION_SIZE):
                    counter += 1
                    dtopic = dicts.UNIT_TOPICS[int(pick(key64(seed, "org.topic2"),
                                                        np.array([counter]),
                                                        len(dicts.UNIT_TOPICS))[0])]
                    div = add(f"Управление {dtopic}", "Управление", 4, dep,
                              bi, ti, ci)
                    for team_size in split(div_size, LEAF_SIZE):
                        counter += 1
                        ttopic = dicts.UNIT_TOPICS[int(pick(key64(seed, "org.topic3"),
                                                            np.array([counter]),
                                                            len(dicts.UNIT_TOPICS))[0])]
                        kd = dicts.UNIT_KINDS[int(pick(key64(seed, "org.kind"),
                                                       np.array([counter]),
                                                       len(dicts.UNIT_KINDS))[0])]
                        add(f"{kd} {ttopic}", kd, 5, div, bi, ti, ci, team_size)

    n = len(name)
    tree = OrgTree(
        unit_id=np.arange(n, dtype=np.int64),
        name=np.array(name, dtype=object),
        kind=np.array(kind, dtype=object),
        level=np.array(level, dtype=np.int64),
        parent=np.array(parent, dtype=np.int64),
        block=np.array(block, dtype=np.int64),
        tb=np.array(tb, dtype=np.int64),
        city=np.array(city, dtype=np.int64),
        headcount=np.array(headcount, dtype=np.int64),
        leaves=np.flatnonzero(np.array(headcount, dtype=np.int64) > 0),
    )
    tree.head_person = np.full(n, -1, dtype=np.int64)
    return tree


def assign_people(tree: OrgTree) -> np.ndarray:
    """Разложить людей по листовым подразделениям.

    Возвращает массив ``unit_id`` длиной в фактическую численность: она равна
    сумме плановых численностей листьев, а не запрошенному ``n``, потому что
    разбиение округляет. Это правильнее подгонки — численность подразделения
    целое число, и никакой мартовский срез не даёт ровно круглый итог.
    """
    return np.repeat(tree.leaves, tree.headcount[tree.leaves])


def unit_external_id(seed: int, unit_id: np.ndarray) -> np.ndarray:
    """Внешний идентификатор подразделения — семизначный, как в ОШС."""
    return 1_000_000 + (integers(key64(seed, "org.extid"), unit_id, 0, 8_999_999))


def city_names() -> list[str]:
    return [c for c, _ in dicts.CITIES]
