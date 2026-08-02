---
name: rows
title: Плоские строки (режим rows)
kind: reference
version: 1.0.0
status: active
domain: general
order: 30
description: Как получить список строк без агрегации и как его отсортировать. Когда нужен перечень, а не число.
tags: [строки, список, rows, сортировка, order_by]
related:
  - {name: getting_started, relation: prerequisite, note: поток запроса}
  - {name: aggregate, relation: alternative, note: если нужно число, а не список}
  - {name: limits, relation: follow_up, note: пагинация}
---

# Плоские строки

Режим включается, когда в теле есть `columns` и нет ни `metrics`, ни
`time_dimensions`.

```json
{
  "schema": "dm_core",
  "logic_model": "employee_actual",
  "columns": [
    "person_id",
    "employee_full_name",
    "grade_level"
  ],
  "filters": {
    "type": "condition",
    "column": "employee_status",
    "operator": "=",
    "value": "активный"
  },
  "order_by": [
    {
      "field": "grade_level",
      "kind": "column",
      "direction": "desc",
      "nulls": "last"
    }
  ],
  "limit": 50
}
```

## Сортировка

`order_by` — список элементов `{field, kind, direction, nulls}`.
В режиме строк сортировать **по метрике нельзя**: метрик в выдаче нет, и
попытка даёт `order-by-metric-not-allowed-in-rows`.

**Указывайте `nulls` явно, если колонка nullable.** По умолчанию пустые
значения идут первыми, и «топ сильнейших» начнётся с тех, у кого показатель не
измерен вовсе.

## limit_by

`limit_by: {limit, by}` даёт N строк на каждую группу — например по три
сотрудника из каждого подразделения. В режиме истории не поддержан.

## Что дальше

- `aggregate` — если нужно число, а не перечень
- `top_n` — ранжированная выборка
- `limits` — что делать, когда строк больше лимита

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
