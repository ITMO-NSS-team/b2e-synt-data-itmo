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
RESOLV_OVERLAY ?= /tmp/b2e-resolv.conf
RESOLV_RECREATE_MARKER ?= var/.resolv-recreate-required

# Single source of truth for the snapshot used by the stand, smoke and
# benchmark. Compose resolves relative DATA_DIR values against deploy/.
DEPLOY_DATA_DIR := $(shell awk -F= '/^DATA_DIR=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
STAND_DATA_DIR := $(or $(DATA_DIR),$(DEPLOY_DATA_DIR),../data)
STAND_DATA_PATH := $(if $(filter /%,$(STAND_DATA_DIR)),$(STAND_DATA_DIR),deploy/$(STAND_DATA_DIR))
SMOKE_DATA ?= $(STAND_DATA_PATH)

.PHONY: help setup catalog data data-small validate stats doc serve test clean \
        up down logs ps seed seed-traps-off smoke check-docs hash-password openapi \
        rebuild sim-test demo benchmarking benchmarking-check benchmarking-smoke \
        benchmarking-generated benchmarking-existing-plus-generated \
        openlit-dashboard prepare-resolv _benchmarking-single

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

prepare-resolv:
	@needs_recreate=""; \
	if test ! -e "$(RESOLV_OVERLAY)"; then \
		needs_recreate=1; \
	fi; \
	if test -d "$(RESOLV_OVERLAY)"; then \
		rmdir "$(RESOLV_OVERLAY)" || { \
			echo "$(RESOLV_OVERLAY) должен быть пустым временным каталогом или файлом"; \
			exit 1; \
		}; \
		needs_recreate=1; \
	fi; \
	test ! -e "$(RESOLV_OVERLAY)" || test -f "$(RESOLV_OVERLAY)" || { \
		echo "$(RESOLV_OVERLAY) существует, но не является обычным файлом"; \
		exit 1; \
	}; \
	printf 'nameserver 127.0.0.11\noptions ndots:0\n' > "$(RESOLV_OVERLAY)"; \
	chmod 644 "$(RESOLV_OVERLAY)"; \
	if test -n "$$needs_recreate"; then \
		mkdir -p "$(dir $(RESOLV_RECREATE_MARKER))"; \
		touch "$(RESOLV_RECREATE_MARKER)"; \
	fi

up: prepare-resolv  ## поднять весь стек одной командой; PROFILE=telegram добавит бота
	@test -f deploy/.env || { echo "нет deploy/.env — скопируйте deploy/.env.example"; exit 1; }
	@mkdir -p var/empty-generated-skills
	@test -d "$(GENERATED_SKILLS_DIR)" || { echo "нет каталога skills: $(GENERATED_SKILLS_DIR)"; exit 1; }
	@force_recreate=""; \
	if test -f "$(RESOLV_RECREATE_MARKER)"; then \
		echo "DNS overlay создан заново; пересоздаю контейнеры с этим bind mount"; \
		force_recreate="--force-recreate"; \
	fi; \
	$(COMPOSE) $(if $(PROFILE),--profile $(PROFILE),) up -d --build $$force_recreate; \
	status=$$?; \
	if test $$status -eq 0; then rm -f "$(RESOLV_RECREATE_MARKER)"; fi; \
	exit $$status
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
DEPLOY_BENCHMARK_REPO_DIR := $(shell awk -F= '/^BENCHMARK_REPO_DIR=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
BENCH_CASES_HOST := $(if $(filter /%,$(CASES)),$(CASES),$(if $(filter b2e-skill-benchmark/%,$(CASES)),$(DEPLOY_BENCHMARK_REPO_DIR)/$(patsubst b2e-skill-benchmark/%,%,$(CASES)),$(abspath $(CASES))))
MAKE_GENERATED_SKILLS_DIR := $(GENERATED_SKILLS_DIR)
DEPLOY_GENERATED_SKILLS_DIR := $(shell awk -F= '/^GENERATED_SKILLS_DIR=/{print substr($$0,index($$0,"=")+1); exit}' deploy/.env 2>/dev/null)
GENERATED_SKILLS_INPUT := $(or $(MAKE_GENERATED_SKILLS_DIR),$(DEPLOY_GENERATED_SKILLS_DIR))
override GENERATED_SKILLS_DIR := $(abspath $(or $(GENERATED_SKILLS_INPUT),var/empty-generated-skills))
export GENERATED_SKILLS_DIR
BENCH_MODES ?= general_knowledge,skills_disabled,existing_skills
BENCH_REPETITIONS ?= 1
BENCH_RESULTS ?= benchmarking/results
BENCH_LIVE_CONFIG ?= var/benchmark-live-config.json
BENCH_EVAL_ID ?=
BENCH_TIMEOUT ?= 1800
BENCH_TRACE_TIMEOUT ?= 300
BENCH_HEALTH_TIMEOUT ?= 60
BENCH_LIMIT ?=
BENCH_REQUIRED_SERVICES := admin-ui b2e-agent phoenix heimdall-emulator
BENCH_EVAL_PREFIX ?= benchmark
BENCH_CHECK_ONLY ?=
BENCH_SKIP_OPENLIT ?=
BENCH_PHASE_TELEMETRY ?=
BENCH_GENERATED_VARIANT_ID ?=
empty :=
space := $(empty) $(empty)
comma := ,
BENCH_MODE_LIST = $(subst $(comma),$(space),$(BENCH_MODES))
BENCH_GENERATED_MODES := generated_skills existing_plus_generated
BENCH_GENERATED_MODE_LIST = $(filter $(BENCH_GENERATED_MODES),$(BENCH_MODE_LIST))
BENCH_GENERATED_MODE_CSV = $(subst $(space),$(comma),$(strip $(BENCH_GENERATED_MODE_LIST)))
BENCH_SKILLS_ROOT = $(if $(BENCH_GENERATED_MODE_LIST),/app/heimdall-skill-catalog/generated,/app/heimdall-skill-catalog/existing)
BENCH_BASELINE_MODE_LIST = $(filter-out $(BENCH_GENERATED_MODES),$(BENCH_MODE_LIST))
BENCH_BASELINE_MODES = $(subst $(space),$(comma),$(strip $(BENCH_BASELINE_MODE_LIST)))

define BENCHMARK_PREPARE
	@$(MAKE) --no-print-directory prepare-resolv
	@mkdir -p var/empty-generated-skills
	@if test -n "$(BENCH_GENERATED_MODE_LIST)" && \
	   printf ',%s,' "$(BENCH_MODES)" | grep -q ',existing_skills,'; then \
		echo "existing_skills и generated-skill режимы используют разные каталоги; запустите их отдельно"; \
		exit 1; \
	fi
	@if test -n "$(BENCH_GENERATED_MODE_LIST)"; then \
		test -n "$(GENERATED_SKILLS_INPUT)" || { echo "для generated_skills задайте GENERATED_SKILLS_DIR"; exit 1; }; \
		test -d "$(GENERATED_SKILLS_DIR)" || { echo "нет каталога generated skills: $(GENERATED_SKILLS_DIR)"; exit 1; }; \
		find "$(GENERATED_SKILLS_DIR)" -type f \( -name '*.md' -o -name '*.yaml' -o -name '*.yml' \) -print -quit | grep -q . || { \
			echo "в $(GENERATED_SKILLS_DIR) нет .md/.yaml skills"; exit 1; \
		}; \
	fi
	@test -f "$(STAND_DATA_PATH)/manifest.json" || { \
		echo "нет $(STAND_DATA_PATH)/manifest.json — создайте/скопируйте снимок или задайте DATA_DIR в deploy/.env"; \
		exit 1; \
	}
	@running="$$($(COMPOSE) ps --status running --services 2>/dev/null)"; \
	missing=""; \
	if test -f "$(RESOLV_RECREATE_MARKER)"; then missing="$$missing resolv-overlay"; fi; \
	for service in $(BENCH_REQUIRED_SERVICES); do \
		printf '%s\n' "$$running" | grep -qx "$$service" || missing="$$missing $$service"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "стенд не готов; запускаю make up (не запущены:$$missing)"; \
		$(MAKE) up GENERATED_SKILLS_DIR="$(GENERATED_SKILLS_DIR)" HEIMDALL_SKILLS_ROOT="$(BENCH_SKILLS_ROOT)"; \
	else \
		echo "стенд поднят; применяю каталог skills: $(BENCH_SKILLS_ROOT)"; \
		HEIMDALL_SKILLS_ROOT="$(BENCH_SKILLS_ROOT)" $(COMPOSE) up -d --no-deps heimdall-emulator; \
		container_id="$$($(COMPOSE) ps -q heimdall-emulator)"; \
		test -n "$$container_id" || { echo "не найден контейнер heimdall-emulator"; exit 1; }; \
		elapsed=0; health=""; \
		while test $$elapsed -lt $(BENCH_HEALTH_TIMEOUT); do \
			health="$$($(DOCKER) inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$$container_id" 2>/dev/null || true)"; \
			if test "$$health" = healthy; then break; fi; \
			if test "$$health" = unhealthy || test "$$health" = exited || test "$$health" = dead; then \
				echo "heimdall-emulator завершил запуск со статусом $$health"; \
				$(COMPOSE) logs --tail=100 heimdall-emulator; \
				exit 1; \
			fi; \
			sleep 1; elapsed=$$((elapsed + 1)); \
		done; \
		if test "$$health" != healthy; then \
			echo "heimdall-emulator не стал healthy за $(BENCH_HEALTH_TIMEOUT) секунд (status=$$health)"; \
			$(COMPOSE) logs --tail=100 heimdall-emulator; \
			exit 1; \
		fi; \
	fi
	@mkdir -p "$(dir $(BENCH_LIVE_CONFIG))" "$(BENCH_RESULTS)" var/benchmark-catalog-snapshots
	$(COMPOSE) exec -T $(if $(BENCH_MODEL),-e B2E_BENCH_MODEL="$(BENCH_MODEL)",) \
		admin-ui python -m sim.benchmark.live_config > "$(BENCH_LIVE_CONFIG)"
endef

openlit-dashboard:  ## создать/обновить dashboard результатов в OpenLIT; без URL безопасно пропускается
	$(PY) -m sim.benchmark.openlit_dashboard --env-file deploy/.env

_benchmarking-single:
	$(BENCHMARK_PREPARE)
	$(if $(or $(BENCH_CHECK_ONLY),$(BENCH_SKIP_OPENLIT)),@true,$(MAKE) --no-print-directory openlit-dashboard)
	$(COMPOSE) --profile benchmark run --rm --no-deps \
		--user "$$(id -u):$$(id -g)" benchmark-runner \
		python3.12 -m sim.benchmark.cli \
			--cases "/app/$(CASES)" --data /data/snapshot \
			--catalog /app/heimdall-skills \
			$(if $(BENCH_GENERATED_MODE_LIST),--generated-skills /app/generated-skills,) \
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
			$(if $(BENCH_PHASE_TELEMETRY),--phase-telemetry,) \
			$(if $(BENCH_GENERATED_VARIANT_ID),--generated-variant-id "$(BENCH_GENERATED_VARIANT_ID)",) \
			$(if $(BENCH_LIMIT),--limit "$(BENCH_LIMIT)",) \
			$(if $(BENCH_EVAL_ID),--eval-id "$(BENCH_EVAL_ID)",)

benchmarking:  ## полный прогон; generated skills изолируются по expected_skills кейса
	@if test -n "$(BENCH_GENERATED_MODE_LIST)"; then \
		set -e; \
		test -n "$(GENERATED_SKILLS_INPUT)" || { \
			echo "для generated_skills задайте непустой GENERATED_SKILLS_DIR"; \
			exit 2; \
		}; \
		eval_id="$(BENCH_EVAL_ID)"; \
		if test -z "$$eval_id"; then eval_id="$(BENCH_EVAL_PREFIX)-$$(date -u +%Y%m%dT%H%M%SZ)"; fi; \
		phase_root="$(BENCH_RESULTS)/.phases/$$eval_id"; \
		plan_root="var/benchmark-generated-plans/$$eval_id"; \
		plan="$$plan_root/plan.json"; \
		phase_list="$$plan_root/phases.txt"; \
		group_list="$$plan_root/groups.tsv"; \
		$(if $(BENCH_CHECK_ONLY),true,$(MAKE) --no-print-directory openlit-dashboard); \
		$(PY) -m sim.benchmark.generated_plan \
			--cases "$(BENCH_CASES_HOST)" \
			--generated-skills "$(GENERATED_SKILLS_DIR)" \
			--base-skills "$(abspath heimdall-skills)" \
			--modes "$(BENCH_GENERATED_MODE_CSV)" \
			--output "$$plan_root" \
			$(if $(BENCH_LIMIT),--limit "$(BENCH_LIMIT)",); \
		covered_cases="$$($(PY) -c 'import json,sys; print(json.load(open(sys.argv[1]))["covered_cases_path"])' "$$plan")"; \
		$(PY) -c 'import json,sys; p=json.load(open(sys.argv[1])); [print("\t".join((g["slug"],g["cases_path"],g["catalog_path"],g["combined_catalog_path"] or "-",",".join(g["skill_names"]),g["variant_id"]))) for g in p["groups"]]' "$$plan" > "$$group_list"; \
		: > "$$phase_list"; \
		if test -n "$(BENCH_BASELINE_MODE_LIST)"; then \
			echo "Baseline-фаза: $(BENCH_BASELINE_MODES) — один раз для покрытых кейсов $$covered_cases"; \
			$(MAKE) --no-print-directory _benchmarking-single \
				CASES="$$covered_cases" \
				BENCH_MODES="$(BENCH_BASELINE_MODES)" BENCH_EVAL_ID="$$eval_id" \
				BENCH_RESULTS="$$phase_root/baseline" BENCH_SKIP_OPENLIT=1 \
				BENCH_PHASE_TELEMETRY=1 \
				BENCH_CHECK_ONLY="$(BENCH_CHECK_ONLY)" BENCH_LIMIT="" \
				BENCH_REPETITIONS="$(BENCH_REPETITIONS)" BENCH_EVAL_PREFIX="$(BENCH_EVAL_PREFIX)"; \
			echo "$$phase_root/baseline/$$eval_id" >> "$$phase_list"; \
		fi; \
		while IFS="$$(printf '\t')" read -r slug cases catalog combined_catalog skill variant <&3; do \
			test -n "$$slug" || continue; \
			for generated_mode in $(BENCH_GENERATED_MODE_LIST); do \
				active_catalog="$$catalog"; \
				if test "$$generated_mode" = "existing_plus_generated"; then active_catalog="$$combined_catalog"; fi; \
				test "$$active_catalog" != "-" || { echo "не собран объединённый каталог для $$variant"; exit 2; }; \
				echo "Generated-фаза: $$generated_mode, $$skill ($$variant), кейсы $$cases"; \
				$(MAKE) --no-print-directory _benchmarking-single \
					CASES="$$cases" GENERATED_SKILLS_DIR="$$active_catalog" \
					BENCH_MODES="$$generated_mode" BENCH_EVAL_ID="$$eval_id" \
					BENCH_GENERATED_VARIANT_ID="$$variant" \
					BENCH_RESULTS="$$phase_root/generated-$$generated_mode-$$slug" BENCH_SKIP_OPENLIT=1 \
					BENCH_PHASE_TELEMETRY=1 \
					BENCH_CHECK_ONLY="$(BENCH_CHECK_ONLY)" BENCH_LIMIT="" \
					BENCH_REPETITIONS="$(BENCH_REPETITIONS)" BENCH_EVAL_PREFIX="$(BENCH_EVAL_PREFIX)"; \
				echo "$$phase_root/generated-$$generated_mode-$$slug/$$eval_id" >> "$$phase_list"; \
			done; \
		done 3< "$$group_list"; \
		merge_args=""; \
		while IFS= read -r phase; do merge_args="$$merge_args --phase $$phase"; done < "$$phase_list"; \
		$(PY) -m sim.benchmark.merge $$merge_args \
			--generated-plan "$$plan" \
			--results "$(BENCH_RESULTS)" --eval-id "$$eval_id"; \
		if test -z "$(BENCH_CHECK_ONLY)"; then \
			$(COMPOSE) --profile benchmark run --rm --no-deps \
				--user "$$(id -u):$$(id -g)" benchmark-runner \
				python3.12 -m sim.benchmark.publish \
				--result-dir "/app/$(BENCH_RESULTS)/$$eval_id"; \
		fi; \
	else \
		$(MAKE) --no-print-directory _benchmarking-single \
			BENCH_MODES="$(BENCH_MODES)" BENCH_CHECK_ONLY="$(BENCH_CHECK_ONLY)" \
			BENCH_LIMIT="$(BENCH_LIMIT)" BENCH_REPETITIONS="$(BENCH_REPETITIONS)" \
			BENCH_EVAL_PREFIX="$(BENCH_EVAL_PREFIX)" BENCH_EVAL_ID="$(BENCH_EVAL_ID)"; \
	fi

benchmarking-check: BENCH_CHECK_ONLY := 1
benchmarking-check: BENCH_EVAL_PREFIX := check
benchmarking-check: benchmarking  ## preflight всех verified-кейсов без вызовов модели

benchmarking-smoke: BENCH_LIMIT := 1
benchmarking-smoke: BENCH_REPETITIONS := 1
benchmarking-smoke: BENCH_EVAL_PREFIX := smoke
benchmarking-smoke: benchmarking  ## первый verified-кейс, один повтор, живой агент

benchmarking-generated: BENCH_MODES := generated_skills
benchmarking-generated: BENCH_EVAL_PREFIX := generated
benchmarking-generated: benchmarking  ## прогон только с каталогом GENERATED_SKILLS_DIR

benchmarking-existing-plus-generated: BENCH_MODES := existing_plus_generated
benchmarking-existing-plus-generated: BENCH_EVAL_PREFIX := existing-plus-generated
benchmarking-existing-plus-generated: benchmarking  ## Heimdall skills плюс один generated skill на фазу
