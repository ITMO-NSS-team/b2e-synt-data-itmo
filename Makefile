PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
export PYTHONPATH := .:skill-factory

OPENAPI ?= Heimdall_openapi.json
HEIMDALL_URL ?= http://127.0.0.1:8081
DATA    ?= data
SEED    ?= 20260801
N       ?= 300000

DOCKER ?= docker
COMPOSE := $(DOCKER) compose -f deploy/docker-compose.yml --env-file deploy/.env
PROFILE ?=

.PHONY: help setup catalog data data-small validate stats doc serve test clean \
        up down logs ps seed seed-traps-off smoke check-docs hash-password openapi \
        rebuild sim-test demo eval-skills eval-deps pin-eval-configs \
	benchmark-data-check benchmark-live-config benchmark-check \
	benchmark-smoke benchmark-run benchmark-server-live-config \
	benchmark-server benchmark-server-smoke

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/' | sort

setup:  ## окружение и зависимости
	python3 -m venv .venv
	.venv/bin/pip install -q -U pip
	.venv/bin/pip install -q numpy pyyaml jsonschema fastapi "uvicorn[standard]" pytest httpx

catalog:  ## каталог витрин из спецификации OpenAPI (OPENAPI=путь)
	$(PY) -m b2e.cli catalog --openapi $(OPENAPI) --overlay catalog --out catalog/snapshot.json

data:  ## полный корпус: 300 000 человек, ~1 ГБ
	$(PY) -m b2e.cli build --seed $(SEED) --n $(N) --out $(DATA)

data-small:  ## быстрый корпус: 3 000 человек, ~9 МБ
	$(PY) -m b2e.cli build --seed $(SEED) --n 3000 --out data-small

validate:  ## гейт согласованности (ненулевой код при отказе)
	$(PY) -m b2e.cli validate --data $(DATA)

stats:  ## отчёт о правдоподобии
	$(PY) -m b2e.cli stats --data $(DATA)

doc:  ## HTML-документация витрин
	$(PY) -m b2e.cli doc --data $(DATA) --out docs/html

serve:  ## эмулятор Heimdall на :8080
	$(PY) -m b2e.cli serve --data $(DATA)

test:  ## тесты; replay-режим, обращений к API модели нет и трат нет
	B2E_LLM_MODE=replay $(PY) -m pytest tests -q

clean:
	rm -rf .pytest-data .pytest_cache **/__pycache__

# ============================================================ симулятор

up:  ## поднять весь стек одной командой; PROFILE=telegram добавит бота
	@test -f deploy/.env || { echo "нет deploy/.env — скопируйте deploy/.env.example"; exit 1; }
	@printf 'nameserver 127.0.0.11\noptions ndots:0\n' > /tmp/b2e-resolv.conf
	@chmod 644 /tmp/b2e-resolv.conf
	$(COMPOSE) $(if $(PROFILE),--profile $(PROFILE),) up -d --build
	@echo "стек поднят; проверка сквозного пути: make smoke"

down:  ## остановить стек, тома сохраняются
	$(COMPOSE) down

logs:  ## логи всех сервисов
	$(COMPOSE) logs -f --tail=100

ps:  ## состояние сервисов и фактическое потребление памяти
	$(COMPOSE) ps
	@docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' | head -20

seed:  ## корпус для стенда: 3 000 человек, ~25 с
	$(PY) -m b2e.cli build --seed $(SEED) --n 3000 --out data-small
	$(PY) -m b2e.cli validate --data data-small

seed-traps-off:  ## корпус без каверз — обязателен для traps_enabled=false (RQ1)
	$(PY) scripts/build_notraps.py --seed $(SEED) --n 3000 --out data-small-notraps

smoke:  ## сквозной путь: вопрос → ответ → трасса → обратная связь → сравнение
	$(PY) scripts/smoke.py

demo:  ## golden path against a running stack (Claude Code / Z.ai, spends plan quota)
	$(PY) scripts/demo.py

check-docs:  ## выполнить каждый пример из heimdall-skills против живого эмулятора
	$(PY) scripts/check_skill_docs.py --url $(HEIMDALL_URL)

hash-password:  ## хэш для BASIC_AUTH_HASH; открытый пароль никуда не пишется
	@docker run --rm -it caddy:2.10-alpine caddy hash-password

openapi:  ## выгрузить OpenAPI b2e-agent и research-api в docs/openapi/
	$(PY) scripts/export_openapi.py

rebuild:  ## пересобрать образы без кэша
	$(COMPOSE) build --no-cache

CATALOG ?= heimdall
LOGGING ?= both
CASE ?= 001-answerable
IGNORE_SNAPSHOT ?= false
PYTHON_IMAGE ?= b2e-itmo/python:local
EVAL_PYTHONPATH := /app:/app/skill-factory:/app/var/eval-site

ifeq ($(CASE),all)
EVAL_CASE := case_ids=null
else
EVAL_CASE := case_ids=[$(CASE)]
endif

eval-deps:  ## hydra-core + mlflow for the eval driver (not the agent image)
	@mkdir -p var/eval-site
	$(DOCKER) run --rm --user root -v $(CURDIR):/app -w /app $(PYTHON_IMAGE) \
		pip install -q --target /app/var/eval-site -r deploy/requirements-eval.txt

pin-eval-configs:  ## append-only clone of agent_config → skills on/off
	$(COMPOSE) up -d --no-deps admin-ui
	$(COMPOSE) exec -T admin-ui python -m sim.skill_eval.pin

eval-skills: pin-eval-configs eval-deps  ## Hydra skill-eval; CATALOG=none|heimdall|factory
	@test -f deploy/.env || { echo "нет deploy/.env"; exit 1; }
	$(DOCKER) run --rm --network host --user root \
		-v $(CURDIR):/app -w /app \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-e PYTHONPATH=$(EVAL_PYTHONPATH) \
		$(PYTHON_IMAGE) \
		python -m sim.skill_eval \
			catalog=$(CATALOG) logging=$(LOGGING) \
			$(EVAL_CASE) ignore_snapshot=$(IGNORE_SNAPSHOT)

# ============================================================ benchmark

CASES ?= benchmarking/cases
# Compose resolves a relative DATA_DIR against deploy/.  Use the same snapshot
# by default so the short benchmark commands cannot silently validate one
# corpus and run against another.  An explicit BENCH_DATA still takes priority.
DEPLOY_DATA_DIR := $(shell awk -F= '/^DATA_DIR=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
BENCH_DATA ?= $(if $(DEPLOY_DATA_DIR),$(if $(filter /%,$(DEPLOY_DATA_DIR)),$(DEPLOY_DATA_DIR),deploy/$(DEPLOY_DATA_DIR)),data-small)
BENCH_MODES ?= general_knowledge,existing_skills
BENCH_REPETITIONS ?= 1
BENCH_RESULTS ?= benchmarking/results
BENCH_LIVE_CONFIG ?= var/benchmark-live-config.json
BENCH_EVAL_ID ?=
BENCH_TIMEOUT ?= 1800
BENCH_TRACE_TIMEOUT ?= 300
BENCH_MODEL ?=
BENCH_LIMIT ?=

benchmark-data-check:  ## убедиться, что локальный снимок данных существует
	@test -f "$(BENCH_DATA)/manifest.json" || { \
		echo "нет $(BENCH_DATA)/manifest.json — сначала выполните make seed или задайте BENCH_DATA"; \
		exit 1; \
	}
BENCH_ARGS = --cases "$(CASES)" --data "$(BENCH_DATA)" \
	--live-config "$(BENCH_LIVE_CONFIG)" --results "$(BENCH_RESULTS)" \
	--modes "$(BENCH_MODES)" --timeout "$(BENCH_TIMEOUT)" \
	--trace-timeout "$(BENCH_TRACE_TIMEOUT)"

benchmark-live-config: export DATA_DIR := $(abspath $(BENCH_DATA))
benchmark-live-config: benchmark-data-check up
	@mkdir -p "$(dir $(BENCH_LIVE_CONFIG))"
	$(COMPOSE) exec -T $(if $(BENCH_MODEL),-e B2E_BENCH_MODEL="$(BENCH_MODEL)",) \
		admin-ui python -m sim.benchmark.live_config > "$(BENCH_LIVE_CONFIG)"

benchmark-check: benchmark-live-config  ## полный preflight всех ready-кейсов, без вызовов агента
	$(PY) -m sim.benchmark.cli $(BENCH_ARGS) --check-only --eval-prefix check \
		$(if $(BENCH_EVAL_ID),--eval-id "$(BENCH_EVAL_ID)",)

benchmark-smoke: benchmark-live-config  ## первый ready-кейс × режимы, один повтор
	$(PY) -m sim.benchmark.cli $(BENCH_ARGS) --limit 1 --repetitions 1 \
		--eval-prefix smoke \
		$(if $(BENCH_EVAL_ID),--eval-id "$(BENCH_EVAL_ID)",)

benchmark-run: benchmark-live-config  ## все ready-кейсы × режимы; BENCH_REPETITIONS=N
	$(PY) -m sim.benchmark.cli $(BENCH_ARGS) \
		--repetitions "$(BENCH_REPETITIONS)" --eval-prefix benchmark \
		$(if $(BENCH_EVAL_ID),--eval-id "$(BENCH_EVAL_ID)",)

benchmark-server-live-config:  ## зафиксировать конфигурацию уже работающего серверного стенда
	@mkdir -p "$(dir $(BENCH_LIVE_CONFIG))" "$(BENCH_RESULTS)" var/benchmark-catalog-snapshots
	$(COMPOSE) exec -T $(if $(BENCH_MODEL),-e B2E_BENCH_MODEL="$(BENCH_MODEL)",) \
		admin-ui python -m sim.benchmark.live_config > "$(BENCH_LIVE_CONFIG)"

benchmark-server: benchmark-server-live-config  ## выполнить benchmark внутри сети серверного стенда
	$(COMPOSE) --profile benchmark run --rm --no-deps \
		--user "$$(id -u):$$(id -g)" benchmark-runner \
		python -m sim.benchmark.cli \
			--cases "/app/$(CASES)" --data /data/snapshot \
			--catalog /app/heimdall-skills \
			--catalog-snapshots /app/var/benchmark-catalog-snapshots \
			--model-catalog /app/catalog/snapshot.json \
			--live-config "/app/$(BENCH_LIVE_CONFIG)" \
			--results "/app/$(BENCH_RESULTS)" \
			--modes "$(BENCH_MODES)" --repetitions "$(BENCH_REPETITIONS)" \
			--timeout "$(BENCH_TIMEOUT)" --trace-timeout "$(BENCH_TRACE_TIMEOUT)" \
			--trace-backend phoenix --agent-url http://b2e-agent:8082 \
			--agent-prefix "" --phoenix-url http://phoenix:6006 --no-auth \
			--eval-prefix server \
			$(if $(BENCH_LIMIT),--limit "$(BENCH_LIMIT)",) \
			$(if $(BENCH_EVAL_ID),--eval-id "$(BENCH_EVAL_ID)",)

benchmark-server-smoke: BENCH_LIMIT := 1
benchmark-server-smoke: BENCH_REPETITIONS := 1
benchmark-server-smoke: benchmark-server  ## первый ready-кейс внутри сети серверного стенда
