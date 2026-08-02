---
name: history
title: "История: окно с шагом периода"
kind: reference
version: 1.0.0
status: active
domain: general
order: 80
description: Как получить динамику по периодам и панель по датам среза. Когда нужно изменение во времени, а не состояние сейчас.
tags: [история, динамика, период, time_dimensions, окно, report_date, carry-forward]
related:
  - {name: getting_started, relation: prerequisite, note: поток запроса}
  - {name: aggregate, relation: prerequisite, note: метрики и группировка}
  - {name: limits, relation: follow_up, note: стоимость запроса}
---

# История

Режим включается наличием `time_dimensions`. Элемент ровно один.

```json
{
  "schema": "dm_core",
  "logic_model": "employee_hist",
  "columns": [
    "company"
  ],
  "metrics": [
    "headcount"
  ],
  "time_dimensions": [
    {
      "name": "report_date",
      "granularity": "month_end",
      "range": {
        "type": "absolute",
        "from": "2026-01-01",
        "to": "2026-04-01"
      },
      "fill": {
        "mode": "previous"
      }
    }
  ]
}
```

## Историю поддерживают не все витрины

| Витрина | Глубина, лет |
|---|---:|
| `anagent.employee_hist_dep` | 5 |
| `dm_core.candidate_hist` | 5 |
| `dm_core.employee` | 5 |
| `dm_core.employee_hist` | 5 |

**Признак берётся из `is_history`, а не из имени.** `position_hist` историю не
поддерживает вопреки суффиксу. Запрос к неисторической витрине даёт
`column-not-a-time-dimension`.

## Окно

`range.type` бывает только `absolute`. **Относительных окон нет**: «за
последние 12 месяцев» через канал не выражается — подставьте конкретные даты.

Окно задаётся как `[from, to)`: правая граница **исключается**. Период, чей
конец совпал с `to`, в выдачу не попадёт.

Гранулярности: `day`, `month_end`, `quarter_end`, `week_end`, `year_end`.
Ширина окна не может превышать глубину истории витрины (5 лет),
иначе `range-exceeds-lookback`.

## С метриками и без

**Без `metrics`** возвращается панель: строка на сущность × период. Значения
переносятся вперёд до следующего изменения (carry-forward). `fill.mode: none`
отключает перенос и оставляет пропуск.

**С `metrics`** возвращается динамика: агрегат по каждому периоду.

## Чего в истории нет

`limit_by` не поддержан. Параметрические колонки и метрики — тоже.
Не дублируйте `report_date` в `columns`: ось и так в выдаче.

## Стоимость

Стоимость примерно равна «периоды × сущности». Дневная гранулярность на
годовом окне — это 365 срезов; сузьте окно или огрубите шаг.

## Смежные темы

- `aggregate` — метрики без временной оси
- `limits` — потолок выдачи и пагинация

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
