"""Колоночное хранилище снимка данных: одна колонка — один файл.

Почему колоночно
----------------
Витрина ``employee_actual`` — 642 колонки. При 3 000 сотрудников это 1,9 млн
ячеек, из которых 63% — массивы. Держать всё в памяти нельзя (машина сборки —
3 ГБ) и не нужно: тело ``mcp_query`` всегда перечисляет конкретные ``columns``.
Ленивая колоночная загрузка совпадает с природой ClickHouse и даёт правдоподобную
стоимость запроса: широкая выборка дороже узкой, и это видно в трейсе.

Формат
------
::

    <root>/<schema>.<model>/
        _rows                    число строк (текст)
        <column>.col             gzip(pickle(list[Any])), имя файла — safe-slug
        _index.json              колонка → имя файла, тип, хэш
    <root>/manifest.json         snapshot_id, хэши всех витрин

Имя файла кодируется, потому что имена колонок содержат точки (``goals.progress``)
и на регистронезависимых ФС различаются только регистром.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_SAFE = re.compile(r"[^a-zA-Z0-9_]")


def encode_column_filename(name: str) -> str:
    """Имя файла для колонки: обратимо и различает регистр."""
    slug = _SAFE.sub("_", name)
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}.{digest}.col"


@dataclass
class TableWriter:
    """Пишет одну витрину колонка за колонкой, не держа её целиком в памяти."""

    root: Path
    key: str
    rows: int = 0
    _index: dict[str, dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.dir = Path(self.root) / self.key
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index = {}

    def write_column(self, name: str, values: list[Any], ch_type: str = "") -> None:
        if self.rows and len(values) != self.rows:
            raise ValueError(
                f"{self.key}.{name}: {len(values)} значений при {self.rows} строках витрины"
            )
        self.rows = self.rows or len(values)
        fname = encode_column_filename(name)
        blob = gzip.compress(pickle.dumps(values, protocol=4), mtime=0)
        (self.dir / fname).write_bytes(blob)
        self._index[name] = {"file": fname, "type": ch_type,
                             "sha256": hashlib.sha256(blob).hexdigest()}

    def close(self) -> str:
        """Записать индекс витрины и вернуть её хэш."""
        (self.dir / "_rows").write_text(str(self.rows), encoding="utf-8")
        payload = json.dumps({"rows": self.rows, "columns": self._index},
                             ensure_ascii=False, indent=1, sort_keys=True)
        (self.dir / "_index.json").write_text(payload, encoding="utf-8")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TableReader:
    """Читает колонки по требованию и кэширует их в пределах процесса."""

    def __init__(self, root: str | Path, key: str, cache_limit: int = 64) -> None:
        self.dir = Path(root) / key
        self.key = key
        self._cache: dict[str, list[Any]] = {}
        self._order: list[str] = []
        self._limit = cache_limit
        idx_path = self.dir / "_index.json"
        if not idx_path.exists():
            raise FileNotFoundError(f"витрина {key} отсутствует в снимке: {self.dir}")
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        self.rows: int = idx["rows"]
        self._index: dict[str, dict] = idx["columns"]

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def available(self) -> set[str]:
        return set(self._index)

    def column(self, name: str) -> list[Any]:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        meta = self._index.get(name)
        if meta is None:
            # Витрина может не материализовать все колонки каталога: то, чего нет
            # в снимке, эквивалентно колонке из NULL — так же, как пустая витрина.
            values = [None] * self.rows
        else:
            values = pickle.loads(gzip.decompress((self.dir / meta["file"]).read_bytes()))
        self._remember(name, values)
        return values

    def columns(self, names: Iterable[str]) -> dict[str, list[Any]]:
        return {n: self.column(n) for n in names}

    def sample(self, name: str, limit: int = 2) -> list[Any]:
        """Первые ``limit`` значений колонки, не заполняя ими кэш.

        Существует ради генератора документации, который просит по паре
        значений у КАЖДОЙ колонки каталога — а их 4599 на 37 витрин. Через
        ``column()`` каждая такая просьба материализует колонку целиком и
        оставляет её в кэше на 64 позиции: на полном корпусе это 12,6 ГиБ и
        SIGKILL от ядра, измерено. Прочитать с диска материализованную колонку
        всё равно приходится целиком — она лежит одним сжатым pickle, — но
        держать её после этого незачем.
        """
        cached = self._cache.get(name)
        if cached is not None:
            return cached[:limit]
        meta = self._index.get(name)
        if meta is None:
            return [None] * min(limit, self.rows)
        values = pickle.loads(gzip.decompress((self.dir / meta["file"]).read_bytes()))
        return values[:limit]

    def _remember(self, name: str, values: list[Any]) -> None:
        self._cache[name] = values
        self._order.append(name)
        while len(self._order) > self._limit:
            self._cache.pop(self._order.pop(0), None)


class Snapshot:
    """Снимок данных целиком: набор витрин плюс манифест с хэшами."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        mpath = self.root / "manifest.json"
        self.manifest: dict = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
        self._readers: dict[str, TableReader] = {}

    @property
    def snapshot_id(self) -> str:
        return self.manifest.get("snapshot_id", "unknown")

    def has(self, key: str) -> bool:
        return (self.root / key / "_index.json").exists()

    def table(self, key: str) -> TableReader:
        r = self._readers.get(key)
        if r is None:
            r = self._readers[key] = TableReader(self.root, key)
        return r

    def rows(self, key: str) -> int:
        """Число строк витрины; 0 — витрина зарегистрирована, но пуста."""
        return self.table(key).rows if self.has(key) else 0

    def keys(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "_index.json").exists())


def write_manifest(root: str | Path, *, seed: int, catalog_sha256: str,
                   tables: dict[str, str], extra: dict | None = None) -> str:
    """Записать манифест снимка и вернуть его ``snapshot_id``.

    ``snapshot_id`` детерминирован: он зависит от seed, версии каталога и хэшей
    витрин, но не от времени сборки — иначе ``manifest_hash`` прогона менялся бы
    на каждой пересборке данных и кэш прогонов обнулялся бы без причины.
    """
    payload = {
        "seed": seed,
        "catalog_sha256": catalog_sha256,
        "tables": dict(sorted(tables.items())),
        **(extra or {}),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload["snapshot_id"] = f"heimdall-sandbox@{digest[:16]}"
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return payload["snapshot_id"]
