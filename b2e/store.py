"""Гибридное хранилище: смысловая поверхность на диске, хвост — на чтении.

Арифметика, из которой следует конструкция
-----------------------------------------
300 000 человек × 4 599 колонок каталога = 1,4 млрд ячеек. При 3 ГБ ОЗУ и 44 ГБ
диска материализовать всё нельзя. Но и обнулять хвост нельзя: агент обязан
искать нужную колонку в реалистично широком каталоге — иначе поиск тривиален и
симуляция ничего не измеряет.

Решение: на диск попадают только колонки, которые цепочка разрешения знает
осмысленно (около 800 из 4 599). Остальные вычисляются при чтении из координаты
``(seed, витрина, колонка, номер строки)`` — детерминированно, поэтому два
воркера API отдают одно и то же, а повторное чтение не меняет ответ.

Формат на диске совпадает с эмулятором Heimdall, поэтому движок запросов
переиспользуется как есть.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from heimdall.catalog import Catalog
from heimdall.store.columnar import TableReader, TableWriter, write_manifest

from b2e.gen import fillers


def _is_array(model, name: str) -> bool:
    member = model.member(name) if model else None
    return bool(member is not None and getattr(member.ch, "is_array", False))


class ProceduralReader(TableReader):
    """Читатель витрины: с диска, а чего нет — из процедурного генератора."""

    def __init__(self, root: str | Path, key: str, catalog: Catalog,
                 seed: int, cache_limit: int = 48) -> None:
        super().__init__(root, key, cache_limit)
        self._catalog = catalog
        self._seed = seed
        self._model = catalog.models.get(key)

    def column(self, name: str) -> list[Any]:
        if name in self._index or self._model is None:
            return super().column(name)
        member = self._model.member(name)
        if member is None or self.rows == 0:
            return [None] * self.rows
        values = fillers.column(name, member.ch, self.key, self._seed,
                                np.arange(self.rows),
                                lengths=self._group_lengths(name, member))
        self._remember(name, values)
        return values

    def sample(self, name: str, limit: int = 2) -> list[Any]:
        """Первые ``limit`` значений, не разыгрывая колонку целиком.

        Здесь выигрыш больше, чем у материализованной колонки, и он же был
        причиной падения. Хвостовых колонок, которых нет на диске, около 3800 на
        витрину — по замыслу, чтобы не материализовать 1,14 млрд ячеек. Но
        ``column()`` разыгрывает их по требованию во всю длину корпуса, так что
        генератор документации, спрашивая по два значения у каждой, просил
        294 000 × 3800 сгенерированных ячеек и получал SIGKILL.

        Заполнитель адресуем: он принимает список индексов строк и считает
        ровно их. Просим два.
        """
        if name in self._index or self._model is None:
            return super().sample(name, limit)
        member = self._model.member(name)
        if member is None or self.rows == 0:
            return [None] * min(limit, self.rows)
        take = min(limit, self.rows)
        lengths = self._group_lengths(name, member, limit=take)
        return fillers.column(name, member.ch, self.key, self._seed,
                              np.arange(take), lengths=lengths)

    def _group_lengths(self, name: str, member,
                       limit: int | None = None) -> np.ndarray | None:
        """Длины массивов вложенной группы — по материализованному соседу.

        У группы вроде ``successors.*`` часть членов лежит на диске (status,
        appoint_date), а часть достаётся заполнителю. Если заполнитель разыграет
        длину сам, i-й элемент его массива будет описывать другого человека,
        чем i-й элемент соседа: именно так ``successors.full_name`` из двух
        элементов оказывался рядом со ``status`` из трёх. Сосед стоит одного
        чтения колонки и не стоит ни байта на диске.
        """
        if "." not in name or not getattr(member.ch, "is_array", False):
            return None
        prefix = name.split(".", 1)[0] + "."
        sibling = next((c for c in sorted(self._index)
                        if c.startswith(prefix) and _is_array(self._model, c)), None)
        if sibling is None:
            return None
        neighbour = (super().sample(sibling, limit) if limit is not None
                     else super().column(sibling))
        return np.array([len(v or []) for v in neighbour], dtype=np.int64)

    def available(self) -> set[str]:
        """Каталожные имена плюс материализованные служебные колонки."""
        declared = set(self._model.columns) if self._model else set()
        return declared | set(self._index)


class ProceduralSnapshot:
    """Снимок целиком. Совместим по интерфейсу с ``heimdall.store.Snapshot``."""

    def __init__(self, root: str | Path, catalog: Catalog | None = None) -> None:
        self.root = Path(root)
        mpath = self.root / "manifest.json"
        self.manifest: dict = (json.loads(mpath.read_text(encoding="utf-8"))
                               if mpath.exists() else {})
        self.seed = int(self.manifest.get("seed", 0))
        self.catalog = catalog or Catalog.load(
            self.manifest.get("catalog_path", "catalog/snapshot.json"))
        self._readers: dict[str, ProceduralReader] = {}

    @property
    def snapshot_id(self) -> str:
        return self.manifest.get("snapshot_id", "unknown")

    def has(self, key: str) -> bool:
        return (self.root / key / "_index.json").exists()

    def table(self, key: str) -> ProceduralReader:
        reader = self._readers.get(key)
        if reader is None:
            reader = self._readers[key] = ProceduralReader(
                self.root, key, self.catalog, self.seed)
        return reader

    def rows(self, key: str) -> int:
        return self.table(key).rows if self.has(key) else 0

    def keys(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "_index.json").exists())


class ChunkedWriter:
    """Запись витрины по частям: колонка целиком в память не поднимается.

    ``TableWriter`` эмулятора принимает колонку одним списком. На 300 000 строк
    массив объектов ещё помещается, а вот собирать все 800 колонок сразу — уже
    нет, поэтому запись идёт колонка за колонкой и список освобождается сразу.
    """

    def __init__(self, root: Path, key: str) -> None:
        self.writer = TableWriter(root, key)

    def write(self, name: str, values: list, ch_type: str = "") -> None:
        self.writer.write_column(name, values, ch_type)

    def close(self) -> str:
        return self.writer.close()


def manifest(root: Path, *, seed: int, catalog: Catalog, tables: dict[str, str],
             extra: dict | None = None) -> str:
    return write_manifest(root, seed=seed, catalog_sha256=catalog.source_sha256,
                          tables=tables, extra=extra or {})
