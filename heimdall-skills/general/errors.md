---
name: errors
title: Коды отказов
kind: reference
version: 1.0.0
status: active
domain: general
order: 120
description: Полная таблица кодов ошибок канала и что делать по каждому. Когда запрос отклонён и надо понять почему.
tags: [ошибка, отказ, код, hint, unknown-column, query-empty]
related:
  - {name: filters, relation: prerequisite, note: грамматика фильтров}
  - {name: rules, relation: prerequisite, note: общие правила канала}
  - {name: limits, relation: alternative, note: лимиты и пагинация}
---

# Коды отказов

Отказ приходит конвертом `{code, detail, hint}`. **Читайте `hint` до того,
как повторять запрос**: в нём написано, что именно исправить.

У ошибок валидации тела дополнительно приходит `errors` — список полей с
причинами.

### HTTP 400

| Код | Что делать |
|---|---|
| `cross-history-join-forbidden` | Соединение исторической модели с неисторической запрещено. |
| `filter-value-invalid` | Значение в фильтре не соответствует типу колонки. Число передавай числом, булево — булевым, дату строкой YYYY-MM-DD. |
| `range-exceeds-lookback` | Окно шире допустимой глубины истории. Сузь диапазон [from, to). |
| `request-validation-error` | Тело не прошло валидацию. Лишние поля запрещены: additionalProperties=false. Смотри errors[] — там loc, msg и type по каждому полю. |
| `unknown-column` | Имя колонки отсутствует в каталоге модели. Возьми точное имя из describe_model; угадывать нельзя. |
| `unknown-metric` | Имя метрики отсутствует в каталоге модели. Метрики и колонки — разные списки в describe_model. |
### HTTP 403

| Код | Что делать |
|---|---|
| `auth-failed` | Заголовок Authorization отсутствует или токен невалиден. Формат: Authorization: Bearer <JWT>. |
| `forbidden` | Нет доступа к модели или инструменту. Проверь action_ids токена и allowed_groups модели. |
### HTTP 404

| Код | Что делать |
|---|---|
| `model-not-found` | Пары schema + logic_model нет в каталоге. Возьми имена из list_models; схема system скрыта и недоступна. |
| `skill-not-found` | Скилла с таким name нет. Найди подходящий через find_skills. |
### HTTP 422

| Код | Что делать |
|---|---|
| `array-join-not-supported-in-history` | Модель-развёртка (ARRAY JOIN) не поддерживает режим истории. |
| `array-operator-requires-array-column` | Операторы has/hasAny/hasAll работают только по Array-колонке. Для скаляра используй condition или condition_in. |
| `column-not-a-time-dimension` | Колонка не объявлена time-dimension. Ось истории — report_date, и только у исторических моделей. |
| `composite-key-arity-mismatch` | Арность кортежа не совпадает с типом колонки. Для emp_key передавай [company, employee_id]. |
| `composite-key-operator-unsupported` | Для колонки Tuple(...) допустимы только =, !=, IN, NOT IN. |
| `granularity-not-allowed` | Гранулярность не входит в column.granularities. Допустимы day, week_end, month_end, quarter_end, year_end. |
| `limit-by-not-supported-in-history` | limit_by в режиме истории не поддержан. Убери его или откажись от time_dimensions. |
| `missing-parameter` | В args не передан объявленный параметр члена. Список параметров — в describe_model, поле parameters. |
| `order-by-kind-mismatch` | Поле order_by.kind не совпадает с классом члена: column для колонки, metric для метрики. |
| `order-by-metric-not-allowed-in-rows` | В режиме строк сортировать по метрике нельзя: метрик в выдаче нет. Добавь metrics (это включит агрегат) или сортируй по колонке. |
| `param-metric-not-supported` | Модель не имеет параметрических метрик. |
| `param-metric-not-supported-in-history` | Параметрические метрики в режиме истории не поддержаны. |
| `param-virtual-not-supported` | Параметрическую колонку нельзя указывать голым именем в columns — передавай её в param_columns как {name, args}. |
| `param-virtual-not-supported-in-history` | Параметрические колонки в режиме истории не поддержаны. |
| `parameter-arity-mismatch` | Число значений аргумента не совпадает с ожидаемым. |
| `parameter-not-literal` | Аргумент параметрического члена должен быть литералом, а не выражением. |
| `parameter-too-many-values` | Слишком много значений в аргументе параметрического члена. |
| `parameter-type-invalid` | Тип значения аргумента не совпадает с объявленным типом параметра. |
| `query-empty` | Не заданы ни columns, ни metrics, ни time_dimensions — нечего возвращать. Режим выводится из структуры тела. |
| `unknown-parameter` | В args передан параметр, которого у члена нет. |
### HTTP 500

| Код | Что делать |
|---|---|
| `internal-error` | Внутренняя ошибка сервиса. Сохрани request_id и повтори позже. |

## Молчаливые отказы

Не всякая проблема даёт код. Ужатие `limit` до потолка, пустая выборка
из-за регистра в фильтре по кириллице и строка `'false'` на булевой колонке
проходят как успех. Если ответ выглядит странно — сверьте типы и посмотрите
на применённый `limit` в ответе.

## Дальше

- `filters` — грамматика условий
- `limits` — потолок и пагинация

Поиск готового рецепта под задачу: `find_skills("<задача>")`. Обзор доменов и приёмов: `get_overview()`.
