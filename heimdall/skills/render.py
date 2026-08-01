"""Сборка markdown, который сервис отдаёт агенту в ``get_skill``.

Почему это отдельный и важный модуль
------------------------------------
Схема ``SkillDetail`` в спеке отдаёт: name, title, kind, version, domain,
description, status, deprecated_by, markdown, related, model, query, variants,
params, output, body.

Полей ``purpose``, ``tags``, ``notes``, ``mandatory_filters``,
``rest_equivalent`` и ``order`` там **нет**. То есть вся императивная часть
рецепта — «меняется только position_id», «пустой ответ означает вакансию»,
«обязательные фильтры не трогай» — доходит до агента исключительно внутри
собранной строки ``markdown``. А это ровно то, ради чего рецепт и существует:
сам запрос агент собрал бы и сам, а вот бизнес-инварианты и семантику пустого
ответа из каталога вывести нельзя.

Точная форма рендера на стороне сервиса нам неизвестна — это открытый вопрос.
Здесь она зафиксирована как допущение и покрыта тестами: если сервис ответит
иначе, разойдётся тест, а не поведение агента в проде.
"""
from __future__ import annotations

import json
from typing import Any


def render(skill: Any) -> str:
    """Собрать markdown скилла."""
    if skill.kind == "reference":
        return _render_reference(skill)
    return _render_recipe(skill)


def _render_reference(skill: Any) -> str:
    parts = [f"# {skill.title}", "", skill.description.strip()]
    if skill.body:
        parts += ["", skill.body.strip()]
    if skill.related:
        parts += ["", "## Связанные скиллы", "", _related_list(skill.related)]
    return "\n".join(parts).strip() + "\n"


def _render_recipe(skill: Any) -> str:
    parts: list[str] = [f"# {skill.title}", "", skill.description.strip()]

    if skill.purpose:
        parts += ["", f"**Назначение.** {skill.purpose.strip()}"]

    if skill.tags:
        # tags тоже нет в SkillDetail. Что с ними делает сервис — открытый
        # вопрос; скорее всего они кормят его индекс релевантности. Рендерим их
        # строкой, чтобы на пути файл → get_skill не терялось ничего: агент
        # выбирает скилл по description, но, уже открыв его, видит и синонимы.
        parts += ["", f"**Ключевые слова.** {', '.join(str(t) for t in skill.tags)}"]

    if skill.model:
        model = skill.model
        parts += ["", "## Модель", "",
                  f"- схема: `{model.get('schema')}`",
                  f"- логическая модель: `{model.get('logic_model')}`"]
        if model.get("table"):
            parts.append(f"- физическая таблица: `{model['table']}` (справочно, для lineage)")

    if skill.mandatory_filters:
        parts += ["", "## Обязательные фильтры", "",
                  "Это бизнес-инварианты выборки. Менять их нельзя: без них ответ будет",
                  "синтаксически верным и содержательно неправильным.", "",
                  "| Колонка | Оператор | Значение | Почему |", "|---|---|---|---|"]
        for f in skill.mandatory_filters:
            parts.append(f"| `{f.get('column')}` | `{f.get('operator')}` | "
                         f"`{f.get('value')}` | {f.get('comment', '')} |")

    if skill.params:
        parts += ["", "## Что подставлять", "",
                  "| Параметр | Куда | Форма | Обязателен | Описание |",
                  "|---|---|---|---|---|"]
        for p in skill.params:
            required = "да" if p.get("required") else "нет"
            parts.append(f"| `{p.get('name')}` | `{p.get('target')}` | "
                         f"{p.get('substitutes', '—')} | {required} | "
                         f"{str(p.get('description', '')).strip()} |")

    if skill.query:
        parts += ["", "## Запрос", "",
                  "Готовое тело `mcp_query` с реальными значениями — исполняется как есть.",
                  "", "```json", json.dumps(skill.query, ensure_ascii=False, indent=2),
                  "```"]

    if skill.variants:
        parts += ["", "## Варианты"]
        for variant in skill.variants:
            parts += ["", f"### {variant.get('title', 'вариант')}"]
            if variant.get("description"):
                parts += ["", str(variant["description"]).strip()]
            parts += ["", "```json",
                      json.dumps(variant.get("query", {}), ensure_ascii=False, indent=2),
                      "```"]
            if variant.get("notes"):
                parts += ["", _as_bullets(variant["notes"])]

    if skill.output:
        parts += ["", "## Что вернётся", "", "| Поле | Что это |", "|---|---|"]
        for field in skill.output:
            parts.append(f"| `{field.get('field')}` | {field.get('title', '')} |")

    if skill.rest_equivalent and skill.model:
        schema, model_name = skill.model.get("schema"), skill.model.get("logic_model")
        parts += ["", "## То же через REST", "",
                  f"Убери `schema` и `logic_model`, тело отправь на "
                  f"`POST /api/v1/{schema}/{model_name}/`."]

    if skill.notes:
        parts += ["", "## Важное", "", _as_bullets(skill.notes)]

    if skill.related:
        parts += ["", "## Связанные скиллы", "", _related_list(skill.related)]

    return "\n".join(parts).strip() + "\n"


def _as_bullets(value: Any) -> str:
    if isinstance(value, str):
        return f"- {value.strip()}"
    return "\n".join(f"- {str(item).strip()}" for item in value)


def _related_list(related: list[dict]) -> str:
    rows = []
    for link in related:
        note = f" — {link['note']}" if link.get("note") else ""
        rows.append(f"- `{link.get('name')}` ({_relation_ru(link.get('relation'))}){note}")
    return "\n".join(rows)


_RELATION_RU = {
    "prerequisite": "сначала",
    "follow_up": "потом",
    "alternative": "вместо",
}


def _relation_ru(relation: str | None) -> str:
    return _RELATION_RU.get(relation or "", relation or "связан")
