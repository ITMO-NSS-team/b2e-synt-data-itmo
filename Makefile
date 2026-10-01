PYTHON := python3.12
PY := $(shell [ -x .venv/bin/python3.12 ] && echo .venv/bin/python3.12 || echo $(PYTHON))
export PYTHONPATH := .:skill-factory

DEPLOY_HARNESS := $(shell awk -F= '/^B2E_HARNESS=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
DEPLOY_MODEL := $(shell awk -F= '/^B2E_MODEL=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
override B2E_HARNESS := $(DEPLOY_HARNESS)
override B2E_MODEL := $(DEPLOY_MODEL)
export B2E_HARNESS B2E_MODEL

OPENAPI ?= Heimdall_openapi.json
HEIMDALL_URL ?= http://127.0.0.1:8081
DATA    ?= data
SEED    ?= 20260801
N       ?= 300000

DOCKER ?= docker
COMPOSE := $(DOCKER) compose -f deploy/docker-compose.yml --env-file deploy/.env
PROFILE ?=

# Single source of truth for the snapshot used by the stand, smoke and
# benchmark. Compose resolves relative DATA_DIR values against deploy/.
DEPLOY_DATA_DIR := $(shell awk -F= '/^DATA_DIR=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
STAND_DATA_DIR := $(or $(DATA_DIR),$(DEPLOY_DATA_DIR),../data)
STAND_DATA_PATH := $(if $(filter /%,$(STAND_DATA_DIR)),$(STAND_DATA_DIR),deploy/$(STAND_DATA_DIR))
SMOKE_DATA ?= $(STAND_DATA_PATH)

.PHONY: help setup catalog data data-small validate stats doc serve test clean \
        up down logs ps seed seed-traps-off smoke check-docs hash-password openapi \
        rebuild sim-test demo benchmarking benchmarking-check benchmarking-smoke \
        openlit-dashboard

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/' | sort

setup:  ## окружение и зависимости
	@command -v $(PYTHON) >/dev/null || { echo "$(PYTHON) is required (macOS: brew install python@3.12)"; exit 1; }
	$(PYTHON) -m venv --clear .venv
	.venv/bin/python3.12 -m pip install -q -U pip
	.venv/bin/python3.12 -m pip install -q -e ".[serve]"

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
	B2E_LLM_MODE=replay B2E_HARNESS=messages_api B2E_MODEL=test-model $(PY) -m pytest tests -q

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
	SMOKE_DATA="$(SMOKE_DATA)" $(PY) scripts/smoke.py

demo:  ## golden path against the selected harness / provider; may spend quota
	$(PY) scripts/demo.py

check-docs:  ## выполнить каждый пример из heimdall-skills против живого эмулятора
	$(PY) scripts/check_skill_docs.py --url $(HEIMDALL_URL)

hash-password:  ## хэш для BASIC_AUTH_HASH; открытый пароль никуда не пишется
	@docker run --rm -it caddy:2.10-alpine caddy hash-password

openapi:  ## выгрузить OpenAPI b2e-agent и research-api в docs/openapi/
	$(PY) scripts/export_openapi.py

rebuild:  ## пересобрать образы без кэша
	$(COMPOSE) build --no-cache

# ============================================================ benchmark

DEPLOY_CASES := $(shell awk -F= '/^CASES=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
CASES ?= $(DEPLOY_CASES)
BENCH_MODES ?= general_knowledge,skills_disabled,existing_skills
BENCH_REPETITIONS ?= 1
BENCH_RESULTS ?= benchmarking/results
BENCH_LIVE_CONFIG ?= var/benchmark-live-config.json
BENCH_EVAL_ID ?=
BENCH_TIMEOUT ?= 1800
BENCH_TRACE_TIMEOUT ?= 300
BENCH_LIMIT ?=
BENCH_REQUIRED_SERVICES := admin-ui b2e-agent phoenix heimdall-emulator
BENCH_EVAL_PREFIX ?= benchmark
BENCH_CHECK_ONLY ?=

define BENCHMARK_PREPARE
	@test -f "$(STAND_DATA_PATH)/manifest.json" || { \
		echo "нет $(STAND_DATA_PATH)/manifest.json — создайте/скопируйте снимок или задайте DATA_DIR в deploy/.env"; \
		exit 1; \
	}
	@running="$$($(COMPOSE) ps --status running --services 2>/dev/null)"; \
	missing=""; \
	for service in $(BENCH_REQUIRED_SERVICES); do \
		printf '%s\n' "$$running" | grep -qx "$$service" || missing="$$missing $$service"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "стенд не готов; запускаю make up (не запущены:$$missing)"; \
		make up; \
	else \
		echo "стенд уже поднят; сервисы не перезапускаются"; \
	fi
	@mkdir -p "$(dir $(BENCH_LIVE_CONFIG))" "$(BENCH_RESULTS)" var/benchmark-catalog-snapshots
	$(COMPOSE) exec -T $(if $(BENCH_MODEL),-e B2E_BENCH_MODEL="$(BENCH_MODEL)",) \
		admin-ui python -m sim.benchmark.live_config > "$(BENCH_LIVE_CONFIG)"
endef

openlit-dashboard:  ## создать/обновить dashboard результатов в OpenLIT; без URL безопасно пропускается
	$(PY) -m sim.benchmark.openlit_dashboard --env-file deploy/.env

benchmarking:  ## полный прогон verified-кейсов внутри сети стенда; BENCH_REPETITIONS=N
	$(BENCHMARK_PREPARE)
	$(if $(BENCH_CHECK_ONLY),@true,$(MAKE) --no-print-directory openlit-dashboard)
	$(COMPOSE) --profile benchmark run --rm --no-deps \
		--user "$$(id -u):$$(id -g)" benchmark-runner \
		python3.12 -m sim.benchmark.cli \
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
			--eval-prefix "$(BENCH_EVAL_PREFIX)" \
			$(if $(BENCH_CHECK_ONLY),--check-only,) \
			$(if $(BENCH_LIMIT),--limit "$(BENCH_LIMIT)",) \
			$(if $(BENCH_EVAL_ID),--eval-id "$(BENCH_EVAL_ID)",)

benchmarking-check: BENCH_CHECK_ONLY := 1
benchmarking-check: BENCH_EVAL_PREFIX := check
benchmarking-check: benchmarking  ## preflight всех verified-кейсов без вызовов модели

benchmarking-smoke: BENCH_LIMIT := 1
benchmarking-smoke: BENCH_REPETITIONS := 1
benchmarking-smoke: BENCH_EVAL_PREFIX := smoke
benchmarking-smoke: benchmarking  ## первый verified-кейс, один повтор, живой агент
