"""Модель каталога витрин: витрина → колонки, метрики, параметрические члены.

Снимок каталога — единственный источник истины и для линтера, и для генератора
данных, и для эмулятора. Он детерминирован: в нём нет ни временных меток, ни
путей рабочей машины, поэтому пересборка на том же входе даёт побайтово тот же
файл, а diff в git показывает изменение каталога, а не шум.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .types import ChType, parse_type

SNAPSHOT_VERSION = "1.0.0"

#: Схема, которую сервис прячет и не даёт запрашивать.
HIDDEN_SCHEMAS = frozenset({"system"})


@dataclass
class Member:
    """Колонка, виртуальная колонка или метрика витрины."""

    name: str
    type: str
    description: str = ""
    #: column | virtual | param_column | metric | param_metric
    kind: str = "column"
    #: Параметры для param_column / param_metric: [{"name":…, "type":…, "description":…}]
    parameters: list[dict] = field(default_factory=list)
    #: Тип элемента массива, если известен (спека его не отдаёт).
    element_type: str | None = None

    _parsed: ChType | None = field(default=None, repr=False, compare=False)

    @property
    def ch(self) -> ChType:
        if self._parsed is None:
            self._parsed = parse_type(self.type)
        return self._parsed

    @property
    def is_parametric(self) -> bool:
        return self.kind in ("param_column", "param_metric")

    @property
    def is_metric(self) -> bool:
        return self.kind in ("metric", "param_metric")

    @property
    def block(self) -> str | None:
        """Имя блока параллельных массивов (``goals.progress`` → ``goals``)."""
        return self.name.split(".", 1)[0] if "." in self.name else None

    def to_json(self) -> dict:
        d = {"name": self.name, "type": self.type, "description": self.description,
             "kind": self.kind}
        if self.parameters:
            d["parameters"] = self.parameters
        if self.element_type:
            d["element_type"] = self.element_type
        return d


@dataclass
class TimeDim:
    name: str
    granularities: list[str]


@dataclass
class Model:
    """Одна логическая витрина каталога."""

    schema: str
    logic_model: str
    summary: str = ""
    table: str | None = None
    rest_path: str | None = None
    deprecated: bool = False
    channels: list[str] = field(default_factory=lambda: ["v2"])
    default_lookback_years: int = 5
    columns: dict[str, Member] = field(default_factory=dict)
    metrics: dict[str, Member] = field(default_factory=dict)
    time_dimensions: dict[str, TimeDim] = field(default_factory=dict)
    #: Имена из enum ``*_Cols``, для которых спека не дала ни типа, ни описания.
    undocumented: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.schema}.{self.logic_model}"

    @property
    def is_history(self) -> bool:
        """История доступна только если у витрины объявлена time-dimension.

        Маршрутизировать по суффиксу имени нельзя: ``position_hist`` историю
        не поддерживает вопреки названию.
        """
        return bool(self.time_dimensions)

    def member(self, name: str) -> Member | None:
        return self.columns.get(name) or self.metrics.get(name)

    def column_names(self) -> set[str]:
        return {n for n, m in self.columns.items() if m.kind == "column"}

    def virtual_names(self) -> set[str]:
        return {n for n, m in self.columns.items() if m.kind == "virtual"}

    def param_column_names(self) -> set[str]:
        return {n for n, m in self.columns.items() if m.kind == "param_column"}

    def metric_names(self) -> set[str]:
        return {n for n, m in self.metrics.items() if m.kind == "metric"}

    def param_metric_names(self) -> set[str]:
        return {n for n, m in self.metrics.items() if m.kind == "param_metric"}

    def selectable_names(self) -> set[str]:
        """Имена, допустимые в ``columns``: обычные и виртуальные, но не параметрические."""
        return self.column_names() | self.virtual_names()

    def blocks(self) -> dict[str, list[str]]:
        """Группы параллельных массивов: ``goals`` → [goals.quarter, goals.title, …]."""
        out: dict[str, list[str]] = {}
        for name, m in self.columns.items():
            if m.block:
                out.setdefault(m.block, []).append(name)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def to_json(self) -> dict:
        return {
            "schema": self.schema,
            "logic_model": self.logic_model,
            "summary": self.summary,
            "table": self.table,
            "rest_path": self.rest_path,
            "deprecated": self.deprecated,
            "channels": self.channels,
            "is_history": self.is_history,
            "default_lookback_years": self.default_lookback_years,
            "columns": [m.to_json() for m in self.columns.values()],
            "metrics": [m.to_json() for m in self.metrics.values()],
            "time_dimensions": [{"name": t.name, "granularities": t.granularities}
                                for t in self.time_dimensions.values()],
            "undocumented": self.undocumented,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        m = cls(
            schema=d["schema"], logic_model=d["logic_model"], summary=d.get("summary", ""),
            table=d.get("table"), rest_path=d.get("rest_path"),
            deprecated=d.get("deprecated", False), channels=d.get("channels", ["v2"]),
            default_lookback_years=d.get("default_lookback_years", 5),
            undocumented=d.get("undocumented", []),
        )
        for c in d.get("columns", []):
            m.columns[c["name"]] = Member(**{k: v for k, v in c.items() if k != "_parsed"})
        for c in d.get("metrics", []):
            m.metrics[c["name"]] = Member(**{k: v for k, v in c.items() if k != "_parsed"})
        for t in d.get("time_dimensions", []):
            m.time_dimensions[t["name"]] = TimeDim(name=t["name"], granularities=t["granularities"])
        return m


@dataclass
class Catalog:
    """Снимок каталога целиком."""

    version: str = SNAPSHOT_VERSION
    source: str = ""
    source_sha256: str = ""
    models: dict[str, Model] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.models.values())

    def __len__(self) -> int:
        return len(self.models)

    def get(self, schema: str, logic_model: str) -> Model | None:
        return self.models.get(f"{schema}.{logic_model}")

    def schemas(self) -> list[str]:
        return sorted({m.schema for m in self.models.values()})

    def visible_in(self, channel: str) -> list[Model]:
        return [m for m in self.models.values() if channel in m.channels]

    def history_models(self) -> list[Model]:
        return [m for m in self.models.values() if m.is_history]

    def counts(self) -> dict[str, int]:
        cols = sum(len(m.columns) for m in self.models.values())
        mets = sum(len(m.metrics) for m in self.models.values())
        return {"models": len(self.models), "schemas": len(self.schemas()),
                "columns": cols, "metrics": mets}

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "counts": self.counts(),
            "models": {k: self.models[k].to_json() for k in sorted(self.models)},
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1, sort_keys=False),
                     encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls(version=d.get("version", SNAPSHOT_VERSION), source=d.get("source", ""),
                source_sha256=d.get("source_sha256", ""))
        for key, md in d.get("models", {}).items():
            c.models[key] = Model.from_json(md)
        return c
