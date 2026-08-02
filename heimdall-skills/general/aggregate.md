---
name: aggregate
title: Агрегаты с группировкой (режим aggregate)
kind: reference
version: 1.0.0
status: active
domain: general
order: 50
description: Как посчитать метрики каталога с группировкой по колонкам. Когда нужен счёт или среднее, а не перечень строк.
tags: [агрегат, группировка, метрика, group by, count]
related:
  - {name: rows, relation: alternative, note: если нужен перечень}
  - {name: param_metrics, relation: follow_up, note: метрики с аргументами}
  - {name: getting_started, relation: prerequisite, note: поток запроса}
---

# Агрегаты с группировкой

Режим включается **наличием `metrics` или `param_metrics`** при отсутствии
`time_dimensions`. Отдельного поля режима нет — это частый источник ошибок:
запрос выглядит агрегатом, а возвращает строки, потому что `metrics` забыли.

```json
{
  "schema": "dm_core",
  "logic_model": "employee_actual",
  "columns": [
    "company"
  ],
  "metrics": [
    "fact_count"
  ],
  "order_by": [
    {
      "field": "fact_count",
      "kind": "metric",
      "direction": "desc"
    }
  ],
  "limit": 50
}
```

`columns` здесь — **ключи группировки**, а не поля выдачи. Без них вернётся
одна строка общего итога.

## Метрики — не колонки

Это разные списки в `describe_model`. Метрику нельзя запросить в `columns`, а
колонку — в `metrics`; и то и другое даёт отказ с понятным кодом.

Примеры метрик витрины `employee_actual`: `agile_ext_emp_count`, `agile_fact_count`, `agile_int_emp_count`, `agile_outstaff_count`, `child_age_max`.

## Ноль и «нет данных» — разные ответы

Счётчик на пустой выборке вернёт `0`, а среднее — `null`. Это не одно и то же:
ноль означает «посчитали, получилось ноль», null — «считать было нечего».

## См. также

- `param_metrics` — метрики, которым нужны аргументы
- `top_n` — ранжирование результата
- `rows` — если агрегат не нужен

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
