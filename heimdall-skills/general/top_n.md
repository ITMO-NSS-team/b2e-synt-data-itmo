---
name: top_n
title: Топ-N и ранжирование
kind: reference
version: 1.0.0
status: active
domain: general
order: 60
description: Как получить N лучших по показателю и не получить вместо них неизмеренных. Когда нужен шорт-лист.
tags: [топ, ранжирование, шортлист, лучшие, order_by, nulls]
related:
  - {name: rows, relation: prerequisite, note: плоские строки и сортировка}
  - {name: aggregate, relation: alternative, note: если ранжируем группы}
  - {name: limits, relation: follow_up, note: потолок выдачи}
---

# Топ-N и ранжирование

Топ собирается сортировкой плюс `limit`. Отдельного механизма ранжирования нет.

```json
{
  "schema": "dm_special",
  "logic_model": "talent_radar_people",
  "columns": [
    "id",
    "full_name",
    "star_index"
  ],
  "filters": {
    "type": "condition_null",
    "column": "star_index",
    "operator": "IS NOT NULL"
  },
  "order_by": [
    {
      "field": "star_index",
      "kind": "column",
      "direction": "desc",
      "nulls": "last"
    }
  ],
  "limit": 10
}
```

## Две ошибки, из-за которых топ получается неверным

**Пустые значения наверху.** Без `nulls: last` неизмеренные окажутся первыми:
в «десятке сильнейших» будут те, кого не оценивали. Либо задавайте `nulls`,
либо отсекайте их фильтром `IS NOT NULL` — как в примере выше.

**Пустое значение — не ноль.** `NULL` в показателе силы означает «не измерен»,
а не «слабый». Схлопывать эти популяции нельзя: если в выборке есть
неизмеренные, скажите, сколько их.

## Топ внутри групп

`limit_by: {limit: 3, by: ["unit_name"]}` даёт по три лучших в каждом
подразделении за один запрос.

## См. также

- `limits` — что делать, когда строк больше потолка
- `compare_people` — сравнение конкретных людей, а не выборки

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
