# Запуск и оценка benchmark-кейсов

Модуль `sim.benchmark` запускает эталонные бизнес-задачи на B2E-стенде и сравнивает ответы агента в режимах с уже существующими навыками информационного сервиса и без них. Единственный путь запуска — одноразовый `benchmark-runner` внутри сети Compose-стенда.

В текущей реализации доступны:

- проверка структуры и условий запуска без обращения к LLM;
- smoke-прогон одного готового кейса;
- полный прогон с несколькими повторениями;
- отдельная сессия агента для каждого запуска;
- сохранение исходных ответов и Phoenix-трасс;
- детерминированная нормализация JSON-ответа;
- расчёт accuracy, инструментальных и ресурсных метрик;
- агрегирование результатов и сравнение режимов.

Generated-режимы запускаются с внешним каталогом, указанным через
`GENERATED_SKILLS_DIR`. Связь кейса со сгенерированным skill задаётся полем
`expected_skills`. Эксперимент ограничивается кейсами, для которых найден хотя бы
один соответствующий артефакт. Baseline-режимы выполняются для каждого такого
кейса ровно один раз, а каждый доступный generated skill проверяется отдельно.
`generated_skills` предоставляет агенту только проверяемый skill;
`existing_plus_generated` — весь стандартный каталог Heimdall и тот же один
generated skill. Несколько generated skills в один каталог не объединяются.
Непокрытые кейсы не становятся фиктивными запусками: их число и причины исключения
фиксируются в generated plan и разделе `selection` итоговой сводки. LLM-as-judge
пока не выполняется — в результатах для него сохраняется только пустая структура.

## Компоненты

| Компонент | Назначение |
| --- | --- |
| `benchmarking/schemas/benchmark-case-v2.schema.json` | JSON Schema авторского кейса |
| `sim/benchmark/cases.py` | загрузка и проверка кейсов |
| `sim/benchmark/contracts.py` | публичный контракт ответа и формирование запроса агенту |
| `sim/benchmark/modes.py` | конфигурации режимов и хеширование каталогов навыков |
| `sim/benchmark/generated_plan.py` | сопоставление кейсов с generated skills и создание изолированных либо расширенных каталогов |
| `sim/benchmark/preflight.py` | проверки перед обращением к LLM |
| `sim/benchmark/execution.py` | запуск изолированной сессии и разбор трассы |
| `sim/benchmark/stand.py` | HTTP-клиент агента и получение Phoenix-трассы |
| `sim/benchmark/pin.py` | append-only конфигурации режимов в Registry |
| `sim/benchmark/trace.py` | извлечение загруженных навыков из трассы |
| `sim/benchmark/scoring.py` | нормализация ответа и расчёт метрик |
| `sim/benchmark/results.py` | сохранение результатов и сводная статистика |
| `sim/benchmark/publish.py` | публикация единой итоговой summary после объединения фаз |
| `sim/benchmark/cli.py` | командный интерфейс benchmark |

## Формат кейса

Каждый кейс хранится отдельным JSON-файлом и соответствует схеме версии `2.0`.

Основные поля:

| Поле | Назначение |
| --- | --- |
| `case_id` | уникальный идентификатор |
| `status` | `draft` или `verified` |
| `category` | категория задачи |
| `query` | исходный пользовательский запрос |
| `gold_contract` | публичная JSON Schema ответа с обязательными `outcome` и `rows` |
| `gold_answer` | скрытый эталонный ответ |
| `gold_comparison` | правила сравнения поля `rows` |
| `employee_id` | пользователь, от имени которого выполняется запрос |
| `employee_role` | ожидаемая роль: `self`, `manager` или `hr` |
| `snapshot_id` | версия набора данных |
| `skill_registry_hash` | версия реестра навыков стенда |
| `expected_skills` | допустимые навыки для решения кейса; каждый доступный generated skill из списка проверяется отдельно |
| `source_type` | тип источника задачи |
| `source_business_process` | исходный бизнес-процесс |

Поддерживаемые категории:

- `answerable`;
- `access_control`;
- `no_data`;
- `missing_skill`;
- `out_of_scope`.

### Статусы

`draft` используется для незавершённых кейсов. В нём могут быть не заполнены `employee_id` и `gold_answer`. При загрузке каталога кейсов такие файлы пропускаются.

`verified` означает, что кейс допускается к preflight и запуску. Для него обязательны:

- существующий `employee_id` с правильной ролью;
- заполненный `gold_contract`;
- заполненные `gold_answer` и `gold_comparison`;
- совместимые `snapshot_id` и `skill_registry_hash`.

Если выбранный путь не содержит ни одного `verified`-кейса, запуск завершается ошибкой.

### Публичный контракт ответа

`gold_contract` содержит форму полного ответа, но не эталонные значения:

```json
{
  "type": "object",
  "required": ["outcome", "rows"],
  "properties": {
    "outcome": {"enum": ["answer", "no_data", "missing_skill", "access_control", "out_of_scope"]},
    "rows": {"type": "array"}
  }
}
```

Тип `rows` доопределяется конкретным кейсом и может быть массивом, объектом, числом или другим JSON-типом. Агент получает только исходный `query` и разрешённый `gold_contract`. Категория кейса, ожидаемый навык, правильный outcome и `gold_answer` в запрос не включаются.

Нормализатор принимает:

- отдельный JSON-объект;
- JSON в Markdown-блоке;
- первый подходящий JSON-объект внутри текстового ответа.

Объект должен соответствовать итоговой JSON Schema. Если подходящий объект не найден, запуск получает статус `normalization_pending`.

### Эталон и правила сравнения

`gold_answer` и `gold_comparison` не передаются агенту:

```json
{
  "outcome": "answer",
  "rows": []
}
```

Правила сравнения хранятся отдельно в `gold_comparison`. Для `answerable` ожидается `answer`. Для `access_control`, `no_data` и `out_of_scope` ожидаемый outcome должен совпадать с категорией. `missing_skill` допускает как успешный ответ базовыми инструментами, так и подтверждённый отказ `missing_skill`.

## Режимы

| Режим | Инструменты | Каталог навыков | Статус |
| --- | --- | --- | --- |
| `general_knowledge` | нет | отсутствует | реализован |
| `skills_disabled` | `list_models`, `describe_model`, `mcp_query` | отсутствует | реализован |
| `existing_skills` | `list_models`, `describe_model`, `get_docs`, `mcp_query`, `find_skills`, `get_skill` | уже существующий каталог навыков информационного сервиса | реализован |
| `generated_skills` | `list_models`, `describe_model`, `mcp_query`, `find_skills`, `get_skill` | только соответствующая кейсу вариация skill из `GENERATED_SKILLS_DIR` | реализован; непокрытые кейсы исключаются из всех выбранных режимов |
| `existing_plus_generated` | `list_models`, `describe_model`, `get_docs`, `mcp_query`, `find_skills`, `get_skill` | стандартные Heimdall skills и ровно одна соответствующая кейсу generated-вариация | реализован; каждый generated skill проверяется отдельной фазой |

В `generated_skills` отсутствует `get_docs`: встроенные процедурные приёмы Heimdall
не должны подменять или дополнять проверяемый сгенерированный skill. Через
`find_skills/get_skill` агент видит только один skill, имя которого указано в
`expected_skills` текущего кейса. Само имя не добавляется в пользовательский
запрос: оно используется только при подготовке каталога экспериментального режима.

В `existing_plus_generated` сохраняются стандартные skills и `get_docs`, а к
каталогу добавляется ровно один проверяемый generated skill. Поэтому режим измеряет,
найдёт ли агент новый skill среди уже существующих и даст ли он преимущество по
сравнению с `existing_skills`. Если в кейсе перечислено несколько
`expected_skills`, planner создаёт отдельный прогон для каждого доступного
generated skill; совместного прогона нескольких generated skills нет.

`general_knowledge` служит отрицательным baseline: агент не получает инструменты стенда, включая `list_models`, `describe_model`, `get_docs`, `mcp_query`, `find_skills` и `get_skill`. Он не видит перечень витрин, не читает данные, не узнаёт поля и метрики и не получает приёмы и рецепты. Режим проверяет, решается ли задача из общих знаний модели без информации со стенда.

`skills_disabled` отделяет доступ к данным от доступа к навыкам. Агент может
найти витрину через `list_models`, узнать её колонки и метрики через
`describe_model` и выполнить запрос через `mcp_query`. Ему недоступны
`find_skills`, `get_skill`, `get_overview` и `get_docs`. Последний также считается
skill-инструментом: он возвращает процедурные приёмы (`aggregate`, `filters`,
`compare_people` и другие), а не бизнес-данные. Каталог навыков к режиму не
подключается, а разрешение на запуск одобренных исполняемых навыков снимается.
Служебный `ToolSearch` остаётся только загрузчиком схем разрешённых data-tools:
MCP-мост публикует ему ровно три инструмента режима и не раскрывает остальные.

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
- статус `verified`;
- совместимость категории и ожидаемого outcome;
- соответствие `gold_answer` публичному контракту;
- наличие snapshot и совпадение `snapshot_id`;
- существование пользователя и соответствие его роли;
- совпадение `skill_registry_hash` с работающим стендом;
- полнота экспериментального fingerprint;
- одинаковые модель, prompt, code policy и conversation mode между режимами;
- загрузка каталога навыков и отсутствие дубликатов;
- неизменность content-addressed snapshot каталога;
- валидность `recipe`, `reference` и исполняемых примеров `mcp_query`;
- существование используемых витрин, полей и метрик.

`make benchmarking-check` выполняет эти проверки без создания агентных сессий и без вызовов LLM.

## Выполнение

Для каждой комбинации `case × repetition × mode` benchmark:

1. проверяет закреплённую конфигурацию режима;
2. формирует сообщение из `query` и `gold_contract`;
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

- заполнен `deploy/.env` для сервисов и `sim/benchmark/.env` для бенчмарка;
- подготовлен снимок данных с `manifest.json` и `truth/people.json`;
- кейсы имеют статус `verified`.

Снимок выбирается только переменной `DATA_DIR` в `deploy/.env` (по умолчанию
`../data`). Тот же путь используется при проверке `manifest.json` и монтируется
в Heimdall и `benchmark-runner` как `/data/snapshot`, поэтому отдельный
`BENCH_DATA_PATH` настраивать не требуется. Разовый запуск можно переопределить
через `DATA_DIR=/absolute/path make benchmarking-check`.

Сервисы используют `deploy/docker-compose.yml` и `deploy/.env`.
Бенчмарк читает собственный `sim/benchmark/.env`; шаблон — `sim/benchmark/.env.example`.
Настройки эксперимента из `deploy/.env` не читаются.
В Compose сохранены исходные `name: b2e-itmo`, имена образов, targets и DNS-overlay.
Локальный `COMPOSE_PROJECT_NAME` в `.env`, если задан, стандартным механизмом
Docker Compose переопределяет имя проекта; перед сменой имени проверьте тома.

```bash
cp sim/benchmark/.env.example sim/benchmark/.env  # один раз, затем заполнить
make setup   # uv создаёт .venv и устанавливает зависимости из pyproject.toml
make up      # исходная команда: подготовка DNS-overlay и up -d --build
```

Benchmark проверяет manifest снимка и состояние нужных сервисов. Ошибка Compose
показывается сразу; она не считается отсутствием стенда. Если сервисов нет или
нужно восстановить DNS-overlay, выполняется исходный `make up`. При готовом
стенде смена каталога skills может пересоздать `heimdall-emulator` без остальных
сервисов. Затем снимается live-config и запускается `benchmark-runner`.

В `sim/benchmark/.env` можно задать `BENCH_MODES`, `BENCH_REPETITIONS`, `BENCH_RESULTS`,
`BENCH_TIMEOUT`, `BENCH_TRACE_TIMEOUT`, `BENCH_HEALTH_TIMEOUT`, `BENCH_LIMIT`
и `BENCH_EVAL_ID`. Аргументы Make имеют приоритет. `BENCH_EVAL_ID` лучше оставить
незаданным, чтобы каждый запуск получил новый идентификатор.
`CASES`, `BENCHMARK_REPO_DIR` и `GENERATED_SKILLS_DIR` в `sim/benchmark/.env`
задаются относительно корня репозитория. `DATA_DIR` остаётся настройкой
стенда в `deploy/.env` и задаётся относительно `deploy/`.
Например: `GENERATED_SKILLS_DIR=generated-skills/all-methods-20261008T113215Z`.
Путь `GENERATED_SKILLS_DIR` в аргументе Make, как и прежде, задаётся относительно
корня проекта либо абсолютным путём.

Локальное окружение устанавливается через `uv pip install -e ".[serve]"`;
`make setup` не пересоздаёт уже существующую `.venv`. `pyproject.toml` и отдельный
`deploy/requirements.txt` сохранены в исходном виде. Синхронизация зависимостей
Docker через lock-файл не вводится. `.dockerignore` исключает локальные окружения,
данные, секреты и результаты из контекста сборки, не меняя Dockerfile.

Если в `sim/benchmark/.env` задан `OPENLIT_CLICKHOUSE_URL`, перед живым прогоном
идемпотентно создаётся или обновляется dashboard `B2E Benchmark`. В нём три
постоянных табличных виджета: список запусков со ссылкой на trace,
агрегированные метрики по режимам и попарные дельты accuracy относительно
`general_knowledge`, `skills_disabled` и `existing_skills`. Абсолютные метрики
предусматривают также `generated_skills` и `existing_plus_generated`, когда
передан внешний каталог.
Колонки сравнения постоянны, а строки создаются только
для режимов, фактически выбранных в конкретном запуске. Виджеты читают
`b2e.benchmark.run` и
`b2e.benchmark.summary.<mode>` из `otel_traces`, поэтому последующие прогоны
появляются после обычного обновления dashboard; создавать его повторно руками
не нужно. `benchmarking-check` не изменяет OpenLIT.

Для локального OpenLIT compose достаточно:

```dotenv
OPENLIT_CLICKHOUSE_URL=http://127.0.0.1:8123
OPENLIT_UI_URL=http://localhost:3000
OPENLIT_DB_USER=default
OPENLIT_DB_PASSWORD=OPENLIT
OPENLIT_DB_NAME=openlit
```

Provisioner можно проверить отдельно командой `make openlit-dashboard`. Если
URL не задан или OpenLIT недоступен, живой benchmark продолжает работу, а в
консоли печатается причина пропуска.

`CASES` может указывать на каталог JSON-файлов, один JSON-файл или JSONL-suite.

### Проверка без LLM

```bash
make benchmarking-check CASES=b2e-skill-benchmark/gold_dataset
```

### Smoke-прогон

Запускает первый `verified`-кейс, три реализованных режима и один повтор:

```bash
make benchmarking-smoke CASES=b2e-skill-benchmark/gold_dataset
```

Smoke проверяет полный технический путь `case → agent → information service → trace → normalization → metrics`. Низкая accuracy не считается технической ошибкой. Ошибки preflight (снимок, роль, каталог, недоступный стенд) останавливают запуск до LLM. После старта хода ненормализуемый JSON, отсутствующая трасса или несовпадение fingerprint не останавливают матрицу: ячейка попадает в `summary.json` → `review`. Ошибка формата учитывается как неверный ответ; отсутствие трассы и несовпадение fingerprint исключаются из quality-метрик. Успешный отказ харнесса (`denied:Bash` и аналоги) при готовом ответе остаётся обычным `completed`.

### Полный прогон

```bash
make benchmarking \
  CASES=b2e-skill-benchmark/gold_dataset \
  BENCH_REPETITIONS=3
```

### Параметры Make

| Параметр | По умолчанию | Назначение |
| --- | --- | --- |
| `DATA_DIR` | значение из `deploy/.env`, иначе `../data` | единственный источник пути к снимку для стенда и benchmark |
| `CASES` | значение `CASES` из `sim/benchmark/.env` | источник кейсов |
| `BENCH_MODES` | `general_knowledge,skills_disabled,existing_skills` | режимы запуска |
| `BENCH_REPETITIONS` | `1` | число повторов полного прогона |
| `BENCH_RESULTS` | `benchmarking/results` | каталог результатов |
| `BENCH_EVAL_ID` | генерируется автоматически | идентификатор запуска |
| `BENCH_TIMEOUT` | `1800` | тайм-аут одного обращения к стенду, секунды |
| `BENCH_TRACE_TIMEOUT` | `300` | максимальное ожидание полной Phoenix-трассы, секунды |
| `BENCH_MODEL` | не задан | model id для benchmark-конфигурации; не меняет provider или harness |
| `BENCH_LIMIT` | не задан | ограничение числа verified-кейсов; без него выполняется весь набор |

## Метрики

Метрики одного запуска имеют значения `0/1`, числовые значения или `null`, если метрика неприменима.

| Метрика | Смысл |
| --- | --- |
| `answer_accuracy` | главная метрика: exact match для задач с ответом или правильность outcome для остальных категорий |
| `strict_answer_accuracy` | та же оценка, но ответ, потребовавший синтаксического восстановления, получает 0 |
| `response_schema_valid_without_repair` | удалось извлечь объект по схеме без восстановления синтаксиса |
| `normalization_repaired` | применено восстановление единственной пропущенной закрывающей скобки корневого объекта |
| `exact_match` | совпадение нормализованного `rows` с `gold_answer.rows` по правилам `gold_comparison` |
| `outcome_accuracy` | совпадение фактического и ожидаемого outcome |
| `correct_refusal` | правильность отказа для `access_control`, `missing_skill` и `out_of_scope` |
| `generated_skill_loaded` | загружен ли через `get_skill` generated skill, одновременно указанный в `case.expected_skills` и входящий в generated-каталог режима; иначе метрика неприменима |
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

Начиная с `benchmark-scorer@2`, после неудачного обычного извлечения JSON
нормализатор может добавить одну завершающую `}` в конце полного ответа или
JSON-блока. Это разрешено только после завершённой строки, массива или объекта,
при ошибке разбора ровно в конце текста и успешной последующей проверке схемы.
Оборванные числа, строки, массивы, отсутствующие поля и неверные типы не
восстанавливаются. Gold-значения в этом исправлении не используются.
`normalization_repair` в записи оценки содержит название применённого правила;
исходный ответ сохраняется без изменений. При сравнении исторических запусков
их сырые ответы следует пересчитать одной версией scorer, сохранив исходные
оценки и отдельно показывая строгую accuracy.

`summary.json` содержит средние значения по режимам и парные сравнения:

- `skills_disabled` относительно `general_knowledge`;
- `existing_skills` относительно `skills_disabled` и `general_knowledge`;
- `generated_skills` относительно `general_knowledge`, когда режим будет подключён;
- `generated_skills` относительно `existing_skills`, когда режим будет подключён.
- `existing_plus_generated` относительно `existing_skills`, когда режим подключён.

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
- `normalization_pending` — ответ получен, но подходящий JSON не извлечён; считается неправильным ответом и требует ручного разбора;
- `condition_invalid` — живой fingerprint не совпал с preflight; ручной разбор, не в средних;
- `unscored` — ход был, но нет трассы, пустой ответ после сбоя клиента и т.п.; ручной разбор, не в средних;
- `draft_skipped` / `mock_skipped` — кейс или режим не допускается к выполнению.

Если выбран generated-режим, причины исключения непокрытых кейсов хранятся не
как статусы запусков, а в `generated_plan.excluded`: отсутствует `expected_skills`
или нет ни одного соответствующего артефакта. Если доступна только часть навыков
из `expected_skills`, кейс запускается отдельно для каждого найденного навыка, а
недоступные имена фиксируются в `generated_plan.partially_covered`.

## Единый запуск

Команды одинаковы для локального и серверного checkout:

```bash
# Первый verified-кейс, один повтор
make benchmarking-smoke CASES=b2e-skill-benchmark/gold_dataset

# Preflight всех verified-кейсов без LLM
make benchmarking-check CASES=b2e-skill-benchmark/gold_dataset

# Полный прогон
make benchmarking \
  CASES=b2e-skill-benchmark/gold_dataset \
  BENCH_REPETITIONS=3

# Generated skill из отдельной директории
make benchmarking-generated \
  GENERATED_SKILLS_DIR=/absolute/path/to/skill-factory/output/skills \
  CASES=b2e-skill-benchmark/gold_dataset

# Стандартный Heimdall-каталог плюс один generated skill на фазу
make benchmarking-existing-plus-generated \
  GENERATED_SKILLS_DIR=/absolute/path/to/skill-factory/output/skills \
  CASES=b2e-skill-benchmark/gold_dataset

# Все пять режимов и один общий отчёт
make benchmarking \
  CASES=b2e-skill-benchmark/gold_dataset \
  BENCH_MODES=general_knowledge,skills_disabled,existing_skills,generated_skills,existing_plus_generated \
  GENERATED_SKILLS_DIR=/absolute/path/to/skill-factory/output/skills
```

`benchmarking` фиксирует live-конфигурацию и **append-only** пинит benchmark-конфиги от текущего `agent_config`, затем запускает одноразовый `benchmark-runner` в сети Compose. Остальные работающие сервисы не пересоздаются, миграции не выполняются, исходные кейсы, snapshot и каталог навыков не изменяются. `heimdall-emulator` пересоздаётся только при изменении подключённой директории skills.

Перед каждым benchmark-запуском Compose применяет выбранное значение
`GENERATED_SKILLS_DIR` к `heimdall-emulator`. Контейнер пересоздаётся только
если его конфигурация монтирования изменилась. Поэтому следующий обычный прогон
без этой переменной снова использует только стандартный каталог.

Если вместе выбраны baseline-режимы и generated-режимы, команда автоматически
делит эксперимент на последовательные фазы. Сначала planner оставляет только
кейсы, для которых найден хотя бы один skill из `expected_skills`. Все
baseline-режимы выполняются по этому общему набору один раз со стандартным
каталогом Heimdall. Затем каждая физическая вариация skill получает отдельную
фазу: изолированный одноэлементный каталог для `generated_skills` и копию
стандартного каталога с этим одним skill для `existing_plus_generated`. Один
skill может обслуживать любое число аугментаций кейса, а несколько вариантов
или имён не размножают baseline-запуски.

Каноническое имя берётся из frontmatter `name`, а идентификатор варианта — из
имени директории артефакта. Например, директории `successors_1/SKILL.md` и
`successors_2/SKILL.md` могут обе содержать `name: successors`. В результатах
они становятся отдельными arms `generated_skills@successors_1` и
`generated_skills@successors_2` и сравниваются с одним набором baseline-ячеек.

Все фазы используют один логический `eval_id`, а затем объединяются в общие
`responses.jsonl`, `scores.jsonl`, `traces/` и `summary.json`. В `summary.json`
`n_runs` включает только реально запущенную матрицу. Раздел `selection` показывает
число исходных, покрытых, выбранных и исключённых кейсов, а также причины
исключения. Попарные дельты считаются только по тем `case_id × repetition`,
которые получили оценку в обоих сравниваемых режимах.
Промежуточные артефакты сохраняются в
`benchmarking/results/.phases/<eval_id>/` для диагностики.
Фазовые score-spans остаются в Phoenix/OpenLIT, но отмечаются как
промежуточные. После merge публикуется ровно одна итоговая сводка;
dashboard игнорирует частичные фазовые summary, чтобы не показывать
последнюю generated-группу как результат всего эксперимента.

Runner обращается напрямую к `http://b2e-agent:8082` и `http://phoenix:6006`. Публичный proxy, HTTP Basic и SSH-транспорт не используются. Результаты сохраняются на той же машине в `benchmarking/results/<eval_id>/`.

Дополнительные параметры:

- `BENCH_LIMIT=5` — выполнить только первые пять verified-кейсов; при наличии
  `generated_skills` лимит применяется после отбора покрытых кейсов;
- `BENCH_RESULTS`, `BENCH_TIMEOUT`, `BENCH_TRACE_TIMEOUT`, `BENCH_EVAL_ID`, `BENCH_REPETITIONS` — параметры запуска и результатов;
- `BENCH_MODES` — по умолчанию `general_knowledge,skills_disabled,existing_skills`;
- `BENCH_MODEL` — модель для pinned benchmark-конфигураций без изменения provider или harness.
- `GENERATED_SKILLS_DIR` — в env путь относительно корня репозитория,
  в аргументе Make — относительно корня репозитория; допустим абсолютный путь к `.md`/`.yaml` skills, созданным генератором.

Если стенд находится на другой машине, сначала нужно войти на неё обычным способом, а затем вызвать одну из трёх команд в серверном checkout.

## Текущие ограничения

- LLM-as-judge не реализован; поле `llm_judge` в score остаётся незаполненным.
- Порядок режимов детерминированный и пока не перемешивается.
- Создание кейсов, получение gold и перевод `draft → verified` выполняются вне runner.
- Серверный прогон требует, чтобы актуальная версия benchmark-кода находилась в checkout стенда.
- `generated_skills` без `GENERATED_SKILLS_DIR` остаётся mock и не выбирается.
- `existing_plus_generated` без `GENERATED_SKILLS_DIR` также остаётся mock и не выбирается.


## Эксперимент по детализации описания

`make benchmarking-detail` сравнивает уже сгенерированные Skill Factory навыки
для `abstract`, `medium` и `detailed`. Артефакты должны лежать в
`<GENERATED_SKILLS_DIR>/<method>/<level>/<artifact>/SKILL.md`.
Генерация не выполняется: сравнивается качество применения готовых скилов.

Для каждой пары метод × детализация создаётся отдельный каталог всех навыков
этой пары, без стандартного каталога Heimdall и без `get_docs`. Агент сам выбирает
нужный скил среди этого каталога; ожидаемый скил не сообщается ему в запросе.
Каждая комбинация должна содержать ровно одну генерацию каждого канонического
навыка. Несколько генераций требуют отдельного эксперимента, а не скрытого выбора.

Все arms используют одинаковые кейсы, покрытые во всех выбранных комбинациях.
Отсутствие любого требуемого скила исключает кейс из всех arms и фиксируется
с причиной. Методы `no-context` и `compact` анализируются раздельно; baseline
`skills_disabled` выполняется один раз по общему набору. Порядок generated-фаз
перемешивается с фиксированным seed. Одновременно выполняются только независимые
stateless-сессии; весь каталог остаётся одним и тем же до конца фазы.

```bash
make benchmarking-detail-check      # preflight всей матрицы, без LLM
make benchmarking-detail-smoke      # один кейс во всех каталогах, живой запуск
make benchmarking-detail            # все verified-кейсы, живой эксперимент

# Только compact; несколько повторов решения каждого кейса
make benchmarking-detail BENCH_DETAIL_METHODS=compact BENCH_REPETITIONS=3

# Продолжить после прерывания без повторного запуска сохранённых ячеек
make benchmarking-detail BENCH_EVAL_ID=detail-<исходный-id> BENCH_RESUME=1
```

Настройки `BENCH_DETAIL_METHODS`, `BENCH_DETAIL_LEVELS`, `BENCH_DETAIL_BASELINES`,
`BENCH_DETAIL_WORKERS`, `BENCH_DETAIL_SEED` можно задать в `sim/benchmark/.env`.
По умолчанию: оба метода; три уровня; один baseline; четыре параллельные сессии;
seed 17. Чтобы убрать baseline, передайте `BENCH_DETAIL_BASELINES=` аргументом Make.
`BENCH_REPETITIONS` — повторы решения, а не повторы генерации.

Один Docker-стенд одновременно обслуживает один benchmark: dispatcher держит
`var/benchmark-stand.lock`. Условия проверяются до первой LLM-сессии каждой фазы
относительно первого manifest; модель, prompt, snapshot, роли и workers должны
совпадать. При resume также должны совпадать входные кейсы, артефакты и настройки.
Сохранённые ответы и оценки не переписываются. Несохранённый ход после прерывания
может потребовать повторного вызова; неполная запись response/score требует разбора.

Дополнительные результаты:

- `detail-report.html` — сводный отчёт для открытия в браузере;
- `detail-report.md` — текстовый отчёт;
- `detail-report.json` — полные агрегаты, разбивка по навыкам и paired comparisons;
- `detail-runs.csv` — каждая ячейка, метрики, session_id, trace_id и путь к трейсу.

Основная quality-метрика — `answer_accuracy`; также доступны exact match,
загрузка ожидаемого generated skill, ошибки и число инструментальных запросов,
токены, задержка и стоимость выполнения. Ошибки формата считаются неверными
ответами; инфраструктурные сбои показаны отдельно и не входят в conditional
accuracy. Парные дельты рассчитаны только на общих scored-ячейках. Наряду с
accuracy по кейсам выводится accuracy с равным весом каждого типа навыка.
95% интервалы для парной macro-дельты получены bootstrap по типам навыков:
аугментации одного навыка не считаются независимыми наблюдениями.

Результат относится к готовым скилам из описаний разной детализации. Он сам по
себе не доказывает необходимость интерактивного этапа уточнения SF: для такого
вывода нужны отдельный контроль с уточнением и без него, несколько генераций
на комбинацию и стоимость самого уточнения. Стоимость генерации из архива
неизвестна; отчёт показывает только стоимость ответов benchmark-агента.


Для быстрого, сбалансированного исследования всех типов навыков:

```bash
make benchmarking-detail BENCH_DETAIL_METHODS=compact BENCH_DETAIL_CASES_PER_SKILL=1
```

Выбор воспроизводим: сначала answerable-кейсы, затем остальные категории, с
сортировкой по case_id. Один тип навыка не получает больше веса из-за большего
числа аугментаций. Такой запуск — пилот; для покрытия всех сценариев нужен полный
набор без `BENCH_DETAIL_CASES_PER_SKILL`.

`brief-summary.md` формируется автоматически вместе с итоговым отчётом.
Для наблюдения за длинным запуском без LLM-вызовов:

```bash
python -m sim.benchmark.detail_progress \
  --plan var/benchmark-generated-plans/<eval_id>/plan.json --watch
```

Страница `<eval_id>-progress.html` располагается рядом с каталогом результатов
и автоматически обновляется. Промежуточные средние по разным фазам не являются
сравнением на одинаковых кейсах; окончательные дельты строятся по matched pairs.
В detail-режиме серия из трёх infrastructure/unscored-сбоев прекращает новые
запросы и сохраняет checkpoint. При изменении live fingerprint запуск также
останавливается. Исправленные read-locks Registry, SkillStore и session Store
позволяют выполнять независимые сессии параллельно, сохраняя read-only границы.

При остановке оператором неполные фазовые результаты сохраняются отдельно в
`benchmarking/results/.phases/<eval_id>/`. Маркер `experiment-status.json` в этом
каталоге со статусом `cancelled` или `interrupted` завершает `detail_progress --watch`;
страница перестаёт обновляться и явно указывает, что эксперимент не завершён.
Исходные `responses.jsonl`, `scores.jsonl` и трейсы при этом не изменяются.
