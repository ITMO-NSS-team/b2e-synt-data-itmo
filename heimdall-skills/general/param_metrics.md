---
name: param_metrics
title: Параметрические метрики и колонки
kind: reference
version: 1.0.0
status: active
domain: general
order: 90
description: Как вызывать члены каталога, которым нужны аргументы. Когда describe_model показал у члена поле parameters.
tags: [параметрические, param_metrics, param_columns, аргументы, distance, вектор]
related:
  - {name: aggregate, relation: prerequisite, note: режим агрегата}
  - {name: filters, relation: prerequisite, note: condition_param}
  - {name: errors, relation: follow_up, note: коды отказов по параметрам}
---

# Параметрические члены

Параметрический член — колонка или метрика, у которой в `describe_model` есть
поле `parameters`. **Голым именем её звать нельзя**: в `columns` она даёт
`param-virtual-not-supported`, в `metrics` — `param-metric-not-supported`.

Правильная форма — отдельные поля `param_columns` и `param_metrics` со списком
объектов `{name, args}`.

```json
{
  "schema": "dm_core",
  "logic_model": "employee_actual",
  "columns": [
    "company"
  ],
  "param_metrics": [
    {
      "name": "oshs_indicator_count",
      "args": {
        "unit_id": 1
      }
    }
  ],
  "limit": 5
}
```

Аргументы берутся из поля `parameters` того же члена в `describe_model`: там
лежат имя, тип и смысл каждого. Лишний аргумент — `unknown-parameter`,
пропущенный — `missing-parameter`.

## Как отличить параметрический член

Только по ответу `describe_model`: в enum имён он лежит рядом с обычными
колонками, и по имени его не отличить.

## Векторный поиск

`distance` у `dm_special.talent_radar_people` — косинусное расстояние до вектора
запроса.

**Размерность `query_vector` обязана совпадать с размерностью `embedding` этой
витрины — 384.** Проверить: `describe_model` показывает её в описании
параметра. Короткий вектор не отклоняется на валидации, а падает уже на
исполнении, с `internal-error` и текстом про несовпадение размерностей — то
есть выглядит как поломка сервиса, а не как ошибка в запросе.

**Обязателен фильтр `has_embedding = true`**, если такая колонка у витрины есть:
иначе `cosineDistance` применится к пустому вектору и запрос упадёт там же.

Пример ниже — иллюстрация формы, а не готовое тело: вместо трёх чисел
подставляется вектор из 384 значений.

```jsonc
{
  "schema": "dm_special",
  "logic_model": "talent_radar_people",
  "columns": ["id", "full_name"],
  "param_columns": [
    { "name": "distance", "args": { "query_vector": [0.01, -0.04, 0.12] } }
  ],
  "filters": { "type": "condition", "column": "has_embedding",
               "operator": "=", "value": true },
  "order_by": [{ "field": "distance", "kind": "column", "direction": "asc" }],
  "limit": 10
}
```

Ближайшие — с наименьшим `distance`, поэтому сортировка по возрастанию.
Не запрашивайте сам `embedding` в `columns`: это сотни чисел на строку.

## В истории не поддержаны

Параметрические члены несовместимы с `time_dimensions` — ни колонки, ни
метрики.

## Отказы по аргументам

`missing-parameter`, `unknown-parameter`, `parameter-type-invalid`,
`parameter-arity-mismatch`, `parameter-too-many-values`, `parameter-not-literal`.
Все читаются одинаково: сверьте `args` со списком `parameters` из
`describe_model`.

## См. также

- `filters` — узел `condition_param` для предиката по такому члену
- `errors` — полная таблица кодов

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
