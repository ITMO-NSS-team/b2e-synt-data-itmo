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
                                np.arange(self.rows))
        self._remember(name, values)
        return values

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
