---
name: getting_started
title: С чего начать первый запрос
kind: reference
version: 1.0.0
status: active
domain: general
order: 10
description: Поток find_skills, describe_model, mcp_query и выбор режима по структуре тела. Когда собираешь первый запрос к Heimdall и не знаешь, с чего начать.
tags: [начало, первый запрос, list_models, describe_model, mcp_query, режим, поток]
related:
  - {name: rows, relation: follow_up, note: режим плоских строк}
  - {name: aggregate, relation: follow_up, note: агрегаты с группировкой}
  - {name: filters, relation: follow_up, note: дерево фильтров}
  - {name: errors, relation: follow_up, note: коды отказов}
---

# Первый запрос к Heimdall

Heimdall отдаёт аналитику поверх ClickHouse через семантический слой. Вы
указываете модель, нужные колонки и метрики, а сервер собирает и исполняет SQL.
Сырой SQL писать не нужно.

## Сначала ищите готовый рецепт

`find_skills("<задача>")` → `get_skill("<имя>")`. Рецепт содержит рабочий
запрос: в нём меняются только поля из `params`, обязательные фильтры не
трогаются. Обзор доменов даёт `get_overview()`.

Ручной путь `list_models` → `describe_model` → `mcp_query` — запасной, для
задач, под которые рецепта ещё нет.

## Размер каталога

Всего **37 витрин** в 7 схемах,
**4599 колонок** и **148 метрик**.

| Схема | Витрин |
|---|---:|
| `anagent` | 7 |
| `dm_core` | 14 |
| `dm_special` | 7 |
| `recruitment` | 4 |
| `sset` | 1 |
| `stable` | 3 |
| `technical` | 1 |

Служебная схема `system` в списке не появляется и для запросов недоступна.

Имена колонок и метрик берутся только из ответа `describe_model`. Угадывать
нельзя: сервер сверяет каждое имя с каталогом и отклоняет незнакомое
(`unknown-column`, `unknown-metric`).

## Режим выводится из структуры тела

Отдельного поля режима нет.

| В теле есть | Режим | Что вернётся |
|---|---|---|
| `time_dimensions` | история | с `metrics` — динамика по периодам; без них — панель |
| `metrics` или `param_metrics` | агрегат | группировка по `columns` |
| только `columns` | строки | плоские строки без агрегации |
| ничего из перечисленного | ошибка | `query-empty` |

## История доступна не у всех

Режим истории поддерживают ровно четыре витрины: `anagent.employee_hist_dep`, `dm_core.candidate_hist`, `dm_core.employee`, `dm_core.employee_hist`.

Признак берётся из поля `is_history` в `list_models`, а НЕ из имени.
`position_hist` историю не поддерживает вопреки суффиксу в названии.

## Первый запрос

```json
{
  "schema": "dm_core",
  "logic_model": "employee_actual",
  "columns": [
    "employee_full_name",
    "grade_level"
  ],
  "limit": 10
}
```

## Куда идти дальше

- `rows` — плоские строки и сортировка
- `aggregate` — метрики и группировка
- `filters` — дерево условий
- `errors` — что означает отказ

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
