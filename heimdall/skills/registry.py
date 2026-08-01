"""Каталог рецептов и приёмов: загрузка с диска, поиск, обзор.

Раскладка — ``<root>/<domain>/<slug>.<ext>``. Домен и имя сервис берёт **из
полей внутри файла**, имя загруженного файла игнорируется; расширение задаёт
вид (``.yaml`` → рецепт, ``.md`` → приём). Мы дополнительно требуем «имя файла
== name», но требует этого линтер, а не загрузчик: загрузчик обязан вести себя
как сервис.

Файл, который не разобрался, **не исчезает молча**: он остаётся в списке с
``ok=False``, как его показывает dev-ручка ``GET /api/v2/dev/skills``. Молча
пропасть — худший вариант: автор считает скилл выложенным, а его нет.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import render as render_module
from .search import build_index

#: Вид скилла определяется расширением файла.
EXTENSIONS = {".yaml": "recipe", ".yml": "recipe", ".md": "reference"}

DEFAULT_LIMIT = 10

#: Текст поля ``note`` в ответе ``get_overview``: сервис прямо разводит понятие
#: со скиллами Anthropic, и это единственное место, где агент об этом узнаёт.
OVERVIEW_NOTE = (
    "Скилл здесь — не «навык агента» в смысле Anthropic Agent Skills, а единица "
    "навигации по каталогу витрин: рецепт (готовый исполняемый запрос) или "
    "справочный приём (документация механизма)."
)

OVERVIEW_INTRO = (
    "Порядок работы: find_skills(«задача») → get_skill(«имя») → mcp_query. "
    "Рецепт содержит рабочий запрос: меняются только поля из params, "
    "обязательные фильтры не трогаются. Если рецепта под задачу нет — "
    "list_models → describe_model → mcp_query."
)

OVERVIEW_FILTERS = (
    "Дерево узлов; у каждого узла поле type, ровно один корень, лишние поля "
    "запрещены. Листья: condition (=,!=,>,<,>=,<=; значение в value) | "
    "condition_in (IN/NOT IN; value — список) | condition_like (LIKE/NOT LIKE/"
    "ILIKE/NOT ILIKE; шаблон в pattern, НЕ в value) | condition_null | "
    "condition_array (has/hasAny/hasAll; только по Array-колонке) | "
    "condition_param (по параметрическому члену: name, args, operator, value). "
    "Подробно — get_skill(\"filters\")."
)


@dataclass
class SkillFile:
    """Файл каталога: разобрался или нет."""

    path: Path
    ok: bool
    name: str | None = None
    error: str | None = None


@dataclass
class Skill:
    """Один скилл каталога."""

    name: str
    title: str
    kind: str
    version: str
    domain: str
    description: str
    status: str = "active"
    deprecated_by: str | None = None
    purpose: str | None = None
    tags: list[str] = field(default_factory=list)
    order: int | None = None
    model: dict | None = None
    rest_equivalent: bool = False
    mandatory_filters: list[dict] = field(default_factory=list)
    params: list[dict] = field(default_factory=list)
    query: dict | None = None
    variants: list[dict] = field(default_factory=list)
    output: list[dict] = field(default_factory=list)
    related: list[dict] = field(default_factory=list)
    notes: list[Any] = field(default_factory=list)
    body: str | None = None
    path: Path | None = None

    @property
    def markdown(self) -> str:
        return render_module.render(self)

    def as_summary(self, relevance: float) -> dict:
        """Карточка выдачи find_skills. Ровно шесть полей, как в спеке."""
        return {"name": self.name, "title": self.title, "domain": self.domain,
                "kind": self.kind, "description": self.description,
                "relevance": relevance}

    def as_overview(self) -> dict:
        return {"name": self.name, "title": self.title, "domain": self.domain,
                "kind": self.kind, "description": self.description}

    def as_detail(self) -> dict:
        """Структурный ответ get_skill. Набор полей фиксирован спекой.

        Обрати внимание, чего здесь НЕТ: purpose, tags, notes, mandatory_filters,
        rest_equivalent, order. Они доходят до агента только внутри markdown.
        """
        return {
            "name": self.name, "title": self.title, "kind": self.kind,
            "version": self.version, "domain": self.domain,
            "description": self.description, "status": self.status,
            "deprecated_by": self.deprecated_by, "markdown": self.markdown,
            "related": self.related, "model": self.model, "query": self.query,
            # SkillVariant в спеке объявляет только title и query — остальное
            # сервер отбрасывает, и полагаться на него в файле нельзя.
            "variants": [{"title": v.get("title", ""), "query": v.get("query", {})}
                         for v in self.variants],
            "params": self.params, "output": self.output, "body": self.body,
        }


class Registry:
    """Каталог скиллов, загруженный с диска."""

    def __init__(self, skills: dict[str, Skill], files: list[SkillFile],
                 duplicates: dict[str, list[str]]) -> None:
        self._skills = skills
        self._files = files
        self._duplicates = duplicates
        self._index = build_index(skills)

    # ------------------------------------------------------------- загрузка

    @classmethod
    def load(cls, root: str | Path) -> "Registry":
        root = Path(root)
        skills: dict[str, Skill] = {}
        files: list[SkillFile] = []
        seen: dict[str, list[str]] = {}

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in EXTENSIONS:
                continue
            try:
                skill = _parse(path)
            except Exception as exc:  # noqa: BLE001 — любой отказ разбора
                files.append(SkillFile(path=path, ok=False, error=str(exc)[:200]))
                continue
            files.append(SkillFile(path=path, ok=True, name=skill.name))
            seen.setdefault(skill.name, []).append(path.name)
            skills[skill.name] = skill

        duplicates = {name: sorted(paths) for name, paths in seen.items() if len(paths) > 1}
        return cls(skills, files, duplicates)

    # -------------------------------------------------------------- доступ

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"скилла {name!r} нет в каталоге")
        return self._skills[name]

    def files(self) -> list[SkillFile]:
        return list(self._files)

    def duplicates(self) -> dict[str, list[str]]:
        return dict(self._duplicates)

    def active(self) -> list[str]:
        return sorted(n for n, s in self._skills.items() if s.status == "active")

    def all_names(self) -> list[str]:
        return sorted(self._skills)

    # --------------------------------------------------------------- поиск

    def find(self, query: str = "*", domain: str | None = None, kind: str | None = None,
             limit: int = DEFAULT_LIMIT, offset: int = 0,
             include_deprecated: bool = False) -> dict:
        names = [n for n, s in self._skills.items()
                 if (include_deprecated or s.status == "active")
                 and (domain is None or domain == "" or s.domain == domain)
                 and (kind is None or kind == "" or s.kind == kind)]

        if query.strip() in ("", "*"):
            scores = {name: 1.0 for name in names}
            ordered = sorted(names)
        else:
            scores = self._index.score(query, names)
            ordered = sorted(names, key=lambda n: (-scores[n], n))

        page = ordered[offset: offset + limit + 1]
        has_next = len(page) > limit
        return {
            "results": [self._skills[n].as_summary(scores[n]) for n in page[:limit]],
            "limit": limit,
            "offset": offset,
            "has_next_page": has_next,
            "hint": None if page else "ничего не нашлось: попробуй find_skills(\"*\")",
        }

    # -------------------------------------------------------------- обзор

    def overview(self) -> dict:
        active = [self._skills[n] for n in self.active()]
        domains: dict[str, int] = {}
        for skill in active:
            domains.setdefault(skill.domain, 0)
            if skill.kind == "recipe":
                domains[skill.domain] += 1
        # Сначала рецепты, потом приёмы — как отдаёт сервис.
        ordered = sorted(active, key=lambda s: (0 if s.kind == "recipe" else 1,
                                                s.domain, s.name))
        return {
            "intro": OVERVIEW_INTRO,
            "domains": [{"name": name, "title": name, "recipe_count": count}
                        for name, count in sorted(domains.items())],
            "skills": [s.as_overview() for s in ordered],
            "filters": OVERVIEW_FILTERS,
            "note": OVERVIEW_NOTE,
        }

    # ---------------------------------------------------------------- граф

    def dangling_links(self) -> dict[str, list[str]]:
        """Ссылки related, указывающие в пустоту: имя → кто на него ссылается."""
        out: dict[str, list[str]] = {}
        known = set(self._skills)
        for skill in self._skills.values():
            for link in skill.related:
                target = link.get("name")
                if target and target not in known:
                    out.setdefault(target, []).append(skill.name)
        return {k: sorted(v) for k, v in sorted(out.items())}


# ------------------------------------------------------------------- разбор

def _parse(path: Path) -> Skill:
    import yaml

    text = path.read_text(encoding="utf-8")
    kind_by_ext = EXTENSIONS[path.suffix]

    if kind_by_ext == "reference":
        meta, body = _split_frontmatter(text)
    else:
        # Рецепт — плоский YAML. Он может начинаться с маркера документа `---`,
        # и это не frontmatter: наивное разбиение по «---» здесь роняло прототип.
        meta, body = yaml.safe_load(text) or {}, None

    if not isinstance(meta, dict):
        raise ValueError("метаданные не разобрались в объект")
    for required in ("name", "title", "kind", "domain", "description"):
        if not meta.get(required):
            raise ValueError(f"нет обязательного поля {required}")
    if meta["kind"] != kind_by_ext:
        raise ValueError(f"kind={meta['kind']} не соответствует расширению {path.suffix}")

    return Skill(
        name=str(meta["name"]),
        title=str(meta["title"]),
        kind=meta["kind"],
        version=str(meta.get("version", "0.0.0")),
        domain=str(meta["domain"]),
        description=" ".join(str(meta["description"]).split()),
        status=str(meta.get("status", "active")),
        deprecated_by=meta.get("deprecated_by"),
        purpose=meta.get("purpose"),
        tags=list(meta.get("tags") or []),
        order=meta.get("order"),
        model=meta.get("model"),
        rest_equivalent=bool(meta.get("rest_equivalent", False)),
        mandatory_filters=list(meta.get("mandatory_filters") or []),
        params=list(meta.get("params") or []),
        query=meta.get("query"),
        variants=list(meta.get("variants") or []),
        output=list(meta.get("output") or []),
        related=list(meta.get("related") or []),
        notes=list(meta.get("notes") or []),
        body=body,
        path=path,
    )


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Разделить YAML-frontmatter и тело приёма."""
    import yaml

    stripped = text.lstrip("﻿")
    if not stripped.startswith("---"):
        raise ValueError("приём обязан начинаться с YAML-frontmatter")
    rest = stripped[3:]
    end = rest.find("\n---")
    if end < 0:
        raise ValueError("frontmatter не закрыт")
    meta = yaml.safe_load(rest[:end]) or {}
    body = rest[end + 4:].lstrip("\n")
    return meta, body
