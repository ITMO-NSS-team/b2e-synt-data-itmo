# Запуск и оценка benchmark-кейсов

Модуль `sim.benchmark` запускает эталонные бизнес-задачи на B2E-стенде и сравнивает ответы агента в режимах с уже существующими навыками информационного сервиса и без них. Runner выполняется рядом со стендом: либо в локальном Compose, либо в одноразовом контейнере внутри сети уже развёрнутого серверного стенда.

В текущей реализации доступны:

- проверка структуры и условий запуска без обращения к LLM;
- smoke-прогон одного готового кейса;
- полный прогон с несколькими повторениями;
- отдельная сессия агента для каждого запуска;
- сохранение исходных ответов и Phoenix-трасс;
- детерминированная нормализация JSON-ответа;
- расчёт accuracy, инструментальных и ресурсных метрик;
- агрегирование результатов и сравнение режимов.

Режим `generated_skills` описан в модели данных, но пока не подключён к CLI: без каталога сгенерированных навыков он считается mock и не запускается. LLM-as-judge также пока не выполняется — в результатах для него сохраняется только пустая структура.

## Компоненты

| Компонент | Назначение |
| --- | --- |
| `benchmarking/schemas/benchmark-case-v3.schema.json` | JSON Schema авторского кейса |
| `sim/benchmark/cases.py` | загрузка и проверка кейсов |
| `sim/benchmark/contracts.py` | публичный контракт ответа и формирование запроса агенту |
| `sim/benchmark/modes.py` | конфигурации режимов и хеширование каталогов навыков |
| `sim/benchmark/preflight.py` | проверки перед обращением к LLM |
| `sim/benchmark/execution.py` | запуск изолированной сессии и разбор трассы |
| `sim/benchmark/scoring.py` | нормализация ответа и расчёт метрик |
| `sim/benchmark/results.py` | сохранение результатов и сводная статистика |
| `sim/benchmark/cli.py` | командный интерфейс benchmark |

## Формат кейса

Каждый кейс хранится отдельным JSON-файлом и соответствует схеме версии `3.0`.

Основные поля:

| Поле | Назначение |
| --- | --- |
| `case_id` | уникальный идентификатор |
| `status` | `draft` или `ready` |
| `category` | категория задачи |
| `query` | исходный пользовательский запрос |
| `response_contract` | публичная JSON Schema результата |
| `evaluation_contract` | скрытый эталон и правила сравнения |
| `employee_id` | пользователь, от имени которого выполняется запрос |
| `employee_role` | ожидаемая роль: `self`, `manager` или `hr` |
| `snapshot_id` | версия набора данных |
| `skill_registry_hash` | версия реестра навыков стенда |
| `expected_skills` | навыки, ожидаемые оценочным контуром |
| `source_type` | тип источника задачи |
| `source_business_process` | исходный бизнес-процесс |

Поддерживаемые категории:

- `answerable`;
- `access_control`;
- `no_data`;
- `missing_skill`;
- `out_of_scope`.

### Статусы

`draft` используется для незавершённых кейсов. В нём могут отсутствовать `employee_id`, `response_contract` и `evaluation_contract`. При загрузке каталога кейсов такие файлы пропускаются.

`ready` означает, что кейс допускается к preflight и запуску. Для него обязательны:

- существующий `employee_id` с правильной ролью;
- заполненный `response_contract`;
- заполненный `evaluation_contract`;
- совместимые `snapshot_id` и `skill_registry_hash`.

Если выбранный путь не содержит ни одного `ready`-кейса, запуск завершается ошибкой.

### Публичный контракт ответа

`response_contract` содержит только форму успешного результата:

```json
{
  "protocol_version": "1.0",
  "result_schema": {
    "type": "array",
    "items": {
      "type": "object"
    }
  }
}
```

Перед запуском benchmark формирует общий конверт:

```json
{
  "result": "значение по result_schema или null",
  "message": "пояснение или null"
}
```

Агент получает только исходный `query` и этот публичный контракт. Категория кейса, ожидаемый навык, правильный outcome и эталонные значения в запрос не включаются.

Нормализатор принимает:

- отдельный JSON-объект;
- JSON в Markdown-блоке;
- первый подходящий JSON-объект внутри текстового ответа.

Объект должен соответствовать итоговой JSON Schema. Если подходящий объект не найден, запуск получает статус `normalization_pending`.

### Эталон и правила сравнения

`evaluation_contract` не передаётся агенту:

```json
{
  "expected_outcome": "answer",
  "gold_result": [],
  "comparison": {
    "ordered": false,
    "row_key": [],
    "allow_extra_rows": false,
    "numeric_absolute_tolerance": 0
  }
}
```

Для `answerable` ожидается `answer`. Для `access_control`, `no_data` и `out_of_scope` ожидаемый outcome должен совпадать с категорией. `missing_skill` допускает как успешный ответ базовыми инструментами, так и подтверждённый отказ `missing_skill`.

## Режимы

| Режим | Инструменты | Каталог навыков | Статус |
| --- | --- | --- | --- |
| `general_knowledge` | нет | отсутствует | реализован |
| `existing_skills` | `list_models`, `describe_model`, `get_docs`, `mcp_query`, `find_skills`, `get_skill` | уже существующий каталог навыков информационного сервиса | реализован |
| `generated_skills` | инструменты режима `existing_skills` | стандартный каталог плюс generated overlay | описан; CLI пока создаёт mock-конфигурацию и отклоняет её выбор |

`general_knowledge` служит отрицательным baseline: агент не получает инструменты стенда, включая `list_models`, `describe_model`, `get_docs`, `mcp_query`, `find_skills` и `get_skill`. Он не видит перечень витрин, не читает данные, не узнаёт поля и метрики и не получает приёмы и рецепты. Режим проверяет, решается ли задача из общих знаний модели без информации со стенда.

Для сравнимости режимов фиксируются одинаковые:

- модель и температура;
- версия системного prompt;
- снимок данных;
- настройки traps и latency;
- список HR-пользователей;
- запрет выполнения произвольного кода;
- stateless-режим диалога.

Меняется только доступность канала навыков и связанный с ним набор инструментов.

## Preflight

Перед первым обращением к модели проверяется вся выбранная матрица `case × mode`.

Проверяются:

- JSON Schema и операционные поля кейса;
- статус `ready`;
- совместимость категории и ожидаемого outcome;
- соответствие `gold_result` публичному контракту;
- наличие snapshot и совпадение `snapshot_id`;
- существование пользователя и соответствие его роли;
- совпадение `skill_registry_hash` с работающим стендом;
- полнота экспериментального fingerprint;
- одинаковые модель, prompt, code policy и conversation mode между режимами;
- загрузка каталога навыков и отсутствие дубликатов;
- неизменность content-addressed snapshot каталога;
- валидность `recipe`, `reference` и исполняемых примеров `mcp_query`;
- существование используемых витрин, полей и метрик.

`benchmark-check` выполняет эти проверки без создания агентных сессий и без вызовов LLM.

Текущая Make-обёртка `benchmark-live-config` рассчитана на Compose-развёртывание на той же машине, где запускается benchmark: она вызывает `make up`, а затем получает фактическую конфигурацию через `docker compose exec admin-ui`. Это ограничение Makefile, а не HTTP-клиента benchmark.

## Выполнение

Для каждой комбинации `case × repetition × mode` benchmark:

1. проверяет закреплённую конфигурацию режима;
2. формирует сообщение из `query` и `response_contract`;
3. создаёт новую stateless-сессию для указанного `employee_id`;
4. выполняет один запрос агенту;
5. получает исходный ответ, статистику и Phoenix-трассу;
6. проверяет, что live fingerprint совпадает с preflight;
7. извлекает JSON-ответ;
8. рассчитывает метрики;
9. сохраняет ответ, score и трассу.

Исходные JSON-файлы кейсов не изменяются.

## Запуск

Требования:

- заполнен `deploy/.env`;
- `PUBLIC_URL` указывает на доступный стенд;
- указан корректный `RESEARCHER_PASSWORD`;
- подготовлен снимок данных с `manifest.json` и `truth/people.json`;
- кейсы имеют статус `ready`.

Команды `benchmark-check`, `benchmark-smoke` и `benchmark-run` самостоятельно поднимают локальный Docker Compose. Для уже работающего серверного стенда используются `benchmark-server` и `benchmark-server-smoke` непосредственно в checkout на сервере.

`CASES` может указывать на каталог JSON-файлов, один JSON-файл или JSONL-suite.

### Проверка без LLM

```bash
make benchmark-check CASES=benchmarking/cases
```

### Smoke-прогон

Запускает первый `ready`-кейс, оба реализованных режима и один повтор:

```bash
make benchmark-smoke CASES=benchmarking/cases
```

Smoke проверяет полный технический путь `case → agent → information service → trace → normalization → metrics`. Низкая accuracy не считается технической ошибкой. Ошибки preflight (снимок, роль, каталог, недоступный стенд) останавливают запуск до LLM. После старта хода ненормализуемый JSON, отсутствующая трасса или несовпадение fingerprint не останавливают матрицу: ячейка попадает в `summary.json` → `review` и не входит в средние метрики. Успешный отказ харнесса (`denied:Bash` и аналоги) при готовом ответе остаётся обычным `completed`.

### Полный прогон

```bash
make benchmark-run \
  CASES=benchmarking/cases \
  BENCH_REPETITIONS=3
```

### Параметры Make

| Параметр | По умолчанию | Назначение |
| --- | --- | --- |
| `CASES` | `benchmarking/cases` | источник кейсов |
| `BENCH_DATA` | `DATA_DIR` из `deploy/.env` или `data-small` | снимок данных, доступный процессу benchmark |
| `BENCH_MODES` | `general_knowledge,existing_skills` | режимы запуска |
| `BENCH_REPETITIONS` | `1` | число повторов полного прогона |
| `BENCH_RESULTS` | `benchmarking/results` | каталог результатов |
| `BENCH_EVAL_ID` | генерируется автоматически | идентификатор запуска |
| `BENCH_TIMEOUT` | `1800` | тайм-аут одного обращения к стенду, секунды |
| `BENCH_TRACE_TIMEOUT` | `300` | максимальное ожидание полной Phoenix-трассы, секунды |
| `BENCH_MODEL` | не задан | model id для benchmark-конфигурации; не меняет provider или harness |
| `BENCH_LIMIT` | не задан | ограничение числа ready-кейсов; без него выполняется весь набор |

## Метрики

Метрики одного запуска имеют значения `0/1`, числовые значения или `null`, если метрика неприменима.

| Метрика | Смысл |
| --- | --- |
| `answer_accuracy` | главная метрика: exact match для задач с ответом или правильность outcome для остальных категорий |
| `exact_match` | совпадение нормализованного `result` с `gold_result` по правилам `comparison` |
| `outcome_accuracy` | совпадение фактического и ожидаемого outcome |
| `correct_refusal` | правильность отказа для `access_control`, `missing_skill` и `out_of_scope` |
| `generated_skill_loaded` | загружен ли целевой generated skill; пока неприменима для стандартных режимов |
| `heimdall_calls` | число обращений к инструментам информационного сервиса (в текущем стенде — Heimdall) |
| `mcp_query_calls` | число вызовов `mcp_query` |
| `failed_tool_calls` | число завершившихся ошибкой инструментальных вызовов |
| `total_tokens` | суммарное число токенов |
| `latency_ms` | полное время выполнения со стороны benchmark-клиента |
| `agent_duration_ms` | длительность корневого span агента |
| `tool_time_ms` | суммарное время инструментальных span |

Для табличных результатов поддерживаются:

- сравнение с учётом или без учёта порядка;
- сопоставление строк по `row_key`;
- разрешение дополнительных строк;
- абсолютная числовая погрешность.

`summary.json` содержит средние значения по режимам и парные сравнения:

- `existing_skills` относительно `general_knowledge`;
- `generated_skills` относительно `general_knowledge`, когда режим будет подключён;
- `generated_skills` относительно `existing_skills`, когда режим будет подключён.

Для метрик сохраняются абсолютная разница и относительное изменение. Для `answer_accuracy` дополнительно рассчитываются разница в процентных пунктах и сокращение ошибки.

## Результаты

Каждый запуск создаёт отдельный каталог:

```text
benchmarking/results/<eval_id>/
├── run-manifest.json
├── preflight.json          # только benchmark-check
├── responses.jsonl         # реальные прогоны
├── scores.jsonl            # реальные прогоны
├── summary.json            # реальные прогоны
└── traces/
    └── <run_id>.json
```

Назначение файлов:

| Файл | Содержимое |
| --- | --- |
| `run-manifest.json` | выбранные кейсы, режимы, повторы и фактическая конфигурация стенда |
| `preflight.json` | результат проверки каждой пары `case × mode` |
| `responses.jsonl` | исходный запрос, ответ, ошибки, observations, session/trace ids и snapshot кейса |
| `scores.jsonl` | нормализованный ответ, метрики и причины ошибок |
| `summary.json` | агрегаты по режимам и относительные сравнения |
| `traces/*.json` | полная трасса конкретного запуска |

Статусы запуска:

- `completed` — ответ нормализован, метрики входят в средние;
- `normalization_pending` — ответ получен, но подходящий JSON не извлечён; ручной разбор, не в средних;
- `condition_invalid` — живой fingerprint не совпал с preflight; ручной разбор, не в средних;
- `unscored` — ход был, но нет трассы, пустой ответ после сбоя клиента и т.п.; ручной разбор, не в средних;
- `draft_skipped` / `mock_skipped` — кейс или режим не допускается к выполнению.

## Запуск на локальном или серверном стенде

### Локальный стенд

Полный локальный путь сам поднимает Compose и затем запускает benchmark-клиент:

```bash
make benchmark-smoke CASES=benchmarking/cases
make benchmark-run CASES=benchmarking/cases BENCH_REPETITIONS=3
```

`benchmark-smoke` выполняет первый ready-кейс в каждом выбранном режиме один раз. `benchmark-run` выполняет весь выбранный набор.

### Уже развёрнутый серверный стенд

Команды выполняются непосредственно в актуальном checkout на сервере:

```bash
cd /var/essdata/b2e-synt-data-itmo

# Первый ready-кейс, один повтор
make benchmark-server-smoke CASES=benchmarking/cases

# Полный прогон
make benchmark-server \
  CASES=benchmarking/cases \
  BENCH_REPETITIONS=3
```

`benchmark-server` фиксирует live-конфигурацию и **append-only** пинит benchmark-конфиги от текущего `agent_config`, затем запускает одноразовый `benchmark-runner` в сети Compose. Долгоживущие сервисы не пересоздаются, миграции не выполняются, исходные кейсы, snapshot и каталог навыков не изменяются.

Runner обращается напрямую к `http://b2e-agent:8082` и `http://phoenix:6006`. Публичный proxy, HTTP Basic и SSH-транспорт не используются. Результаты сохраняются на той же машине в `benchmarking/results/<eval_id>/`.

Дополнительные параметры:

- `BENCH_LIMIT=5` — выполнить только первые пять ready-кейсов;
- `BENCH_RESULTS`, `BENCH_TIMEOUT`, `BENCH_TRACE_TIMEOUT`, `BENCH_EVAL_ID`, `BENCH_REPETITIONS` — параметры запуска и результатов;
- `BENCH_MODES` — по умолчанию `general_knowledge,existing_skills`;
- `BENCH_MODEL` — модель для pinned benchmark-конфигураций без изменения provider или harness.

Отдельного remote-driver нет. Если стенд находится на другой машине, сначала нужно войти на неё обычным способом, а затем вызвать `benchmark-server` или `benchmark-server-smoke` в серверном checkout.

## Текущие ограничения

- CLI запускает `general_knowledge` и `existing_skills`; `generated_skills` запускается после подключения generated overlay.
- `generated_skills` требует отдельного механизма подключения combined-каталога.
- LLM-as-judge не реализован; поле `llm_judge` в score остаётся незаполненным.
- Порядок режимов детерминированный и пока не перемешивается.
- Создание кейсов, получение gold и перевод `draft → ready` выполняются вне runner.
- Серверный прогон требует, чтобы актуальная версия benchmark-кода находилась в checkout стенда.
- `generated_skills` остаётся mock до подключения combined-каталога к server runner.
