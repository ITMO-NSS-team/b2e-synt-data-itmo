---
name: compare_people
title: Сравнение конкретных людей
kind: reference
version: 1.0.0
status: active
domain: general
order: 70
description: Как получить одинаковый срез по нескольким сотрудникам одним запросом. Когда нужно сравнить двоих или троих.
tags: [сравнение, сравнить, несколько сотрудников, condition_in, составной ключ, emp_key]
related:
  - {name: rows, relation: prerequisite, note: плоские строки}
  - {name: filters, relation: prerequisite, note: дерево фильтров}
  - {name: rules, relation: follow_up, note: границы канала}
---

# Сравнение конкретных людей

Сравнение — это один запрос со списком идентификаторов, а не несколько
запросов подряд. Так гарантируется одинаковый срез и исключается путаница
между людьми.

```json
{
  "schema": "dm_core",
  "logic_model": "employee_actual",
  "columns": [
    "person_id",
    "employee_full_name",
    "grade_level",
    "estimation_year",
    "mean_value_completion"
  ],
  "filters": {
    "type": "condition_in",
    "column": "person_id",
    "operator": "IN",
    "value": [
      "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "a47ac10b-58cc-4372-a567-0e02b2c3d480"
    ]
  },
  "limit": 10
}
```

## Составной ключ

Если людей адресуют парой «компания и табельный номер», это `emp_key` типа
`Tuple(String, String)`. Список кортежей передаётся так:

```json
{
  "type": "condition_in",
  "column": "emp_key",
  "operator": "IN",
  "value": [
    [
      "paosberbank",
      "1536556"
    ],
    [
      "paosberbank",
      "1536557"
    ]
  ]
}
```

Арность обязана совпадать с типом колонки: одиночное значение вместо пары
даёт `composite-key-arity-mismatch`.

## Чего делать не нужно

**Не сравнивайте по разным срезам.** Два отдельных запроса могут прийтись на
разные состояния данных; один запрос с `IN` этого лишён.

**Не считайте разницу в запросе.** Произвольных вычислений в канале нет —
сравнение и разности считаются в пост-обработке.

## См. также

- `filters` — полная грамматика условий
- `rules` — что через канал невозможно

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
