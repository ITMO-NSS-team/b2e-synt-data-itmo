# Бенчмарк

Параметры эксперимента задаются в `sim/benchmark/.env` (шаблон — `.env.example`).
`deploy/.env` — единственный источник настроек стенда: снимок `DATA_DIR`,
модель `B2E_MODEL`, ключи, Phoenix, публичный адрес и подключение к OpenLIT.
Их не нужно повторять в benchmark-файле: `make benchmarking*` использует
конфигурацию работающих сервисов, а dashboard читает `deploy/.env`.
Локальный `sim/benchmark/.env` исключён из Git и Docker-контекста.
Для `make benchmarking*` значения из него имеют приоритет над старыми
переменными, экспортированными в терминале.

```bash
test -f sim/benchmark/.env || cp sim/benchmark/.env.example sim/benchmark/.env
# Заполнить CASES, BENCHMARK_REPO_DIR, BENCH_MODES и при необходимости
# BENCH_SKILL_CATALOGS и остальные параметры в sim/benchmark/.env.
make benchmarking-check
make benchmarking
```

Команды вызываются из корня стенда без передачи параметров в CLI. Относительные
пути к кейсам и каталогам skills в `sim/benchmark/.env` отсчитываются от корня
стенда. Для отдельного репозитория кейсов укажите
`BENCHMARK_REPO_DIR=../b2e-skill-benchmark` и
`CASES=b2e-skill-benchmark/gold_dataset`. Для generated-режимов добавьте, например,
`BENCH_SKILL_CATALOGS=baseline=/abs/run,skillfactory=/abs/output` и перечислите
нужные режимы в `BENCH_MODES`. Для нового многокаталожного запуска с
generated-режимами `BENCH_SKILL_CATALOGS` должен быть заполнен.
Каждый готовый skill-файл из этих каталогов получает отдельный вариант прогона;
каталог может содержать вложенные папки или плоский набор файлов. Название и
версия skill берутся из его содержимого. Варианты без соответствующих выбранных
кейсов показываются в отчёте как непокрытые; каталоги для них не копируются и
агент не вызывается. При выборе generated-режима все остальные выбранные режимы
выполняются только на кейсах, которым соответствует хотя бы один generated skill.
Если совпадений нет совсем, создаётся отчёт с нулём запусков и причинами
исключения кейсов.
Пути результатов `BENCH_RESULTS` отсчитываются от корня стенда.
`CASES` и `BENCHMARK_REPO_DIR` задавайте только здесь, не в `deploy/.env`.

`make benchmarking-check` выполняет preflight без вызова модели;
`make benchmarking-smoke` запускает первый подходящий кейс один раз;
`make benchmarking` запускает весь выбранный эксперимент. Чтобы продолжить
прерванный прогон, укажите прежний `BENCH_EVAL_ID` и `BENCH_RESUME=1` в
`sim/benchmark/.env`; для следующего нового прогона удалите эти значения.

Описание режимов, метрик и формата результатов:
[benchmark-execution-description.md](../../docs/benchmark-execution-description.md).
