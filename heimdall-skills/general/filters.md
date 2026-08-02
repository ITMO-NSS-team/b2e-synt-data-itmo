---
name: filters
title: Дерево фильтров
kind: reference
version: 1.0.0
status: active
domain: general
order: 15
description: Все типы узлов фильтра, их поля и операторы, включая condition_param. Когда нужно ограничить выборку условием.
tags: [фильтр, условие, where, condition, condition_param, массив, составной ключ]
related:
  - {name: getting_started, relation: prerequisite, note: поток запроса}
  - {name: rules, relation: follow_up, note: общие правила канала}
  - {name: errors, relation: follow_up, note: коды отказов}
---

# Дерево фильтров

`filters` — это **одно дерево с ровно одним корнем**, а не список условий.
У каждого узла есть поле `type`. Лишнее поле отвергает **всё тело целиком**
(`additionalProperties: false`), а не только этот узел.

## Листья дерева

| Тип узла | Поля | Операторы |
|---|---|---|
| `condition` | `cast`, `column`, `is_tuple`, `operator`, `value` | `!=`, `<`, `<=`, `=`, `>`, `>=` |
| `condition_in` | `column`, `is_tuple`, `operator`, `value` | `IN`, `NOT IN` |
| `condition_like` | `case_sensitive`, `column`, `operator`, `pattern` | `ILIKE`, `LIKE`, `NOT ILIKE`, `NOT LIKE` |
| `condition_null` | `column`, `operator` | `IS NOT NULL`, `IS NULL` |
| `condition_array` | `cast`, `column`, `flatten`, `operator`, `value` | `has`, `hasAll`, `hasAny` |
| `condition_param` | `args`, `expr`, `name`, `operator`, `value` | `!=`, `<`, `<=`, `=`, `>`, `>=` |

## Логические узлы

- `and` / `or` — объединяют несколько условий: `{"type": "and", "conditions": [...]}`,
  список непустой;
- `not` — инвертирует одно: `{"type": "not", "condition": {...}}`.

## Что чаще всего ломается

**Шаблон у `condition_like` лежит в `pattern`, а не в `value`.** Это самая
частая ошибка: тело с `value` отвергается целиком, а не игнорируется.

```json
{
  "type": "condition_like",
  "column": "employee_full_name",
  "operator": "ILIKE",
  "pattern": "%иванов%"
}
```

**Array-колонку фильтруют только через `condition_array`.** Операторы `has`,
`hasAny`, `hasAll` сравнивают элемент целиком — подстроки здесь нет. Для
`has` значение скалярное, для остальных — непустой список.

**Составной ключ `emp_key` — это `Tuple(company, employee_id)`.** Допустимы
только `=`, `!=`, `IN`, `NOT IN`; значение передаётся массивом, и арность
обязана совпадать.

```json
{
  "type": "condition",
  "column": "emp_key",
  "operator": "=",
  "value": [
    "paosberbank",
    "1536556"
  ]
}
```

**Параметрическую колонку адресуют узлом `condition_param` полем `name`,
а не `column`.** Аргументы члена идут в `args`.

```json
{
  "type": "condition_param",
  "name": "oshs_owned",
  "args": {
    "unit_id": 2120663
  },
  "operator": "=",
  "value": 1
}
```

По параметрическим **метрикам** в `filters` фильтровать нельзя: это агрегат.

**Кириллица и регистр.** ClickHouse `lower()` и `ILIKE` не сворачивают
кириллицу. Фильтр `%петров%` не найдёт запись `ПЕТРОВ ПЁТР` и вернёт пусто
без единой ошибки.

## Связанные скиллы

- `rules` — общие правила канала
- `errors` — что означает каждый код отказа

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
