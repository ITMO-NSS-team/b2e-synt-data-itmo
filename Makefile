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
        rebuild sim-test demo eval-skills eval-deps pin-eval-configs

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/' | sort

setup:  ## окружение и зависимости
	python3 -m venv .venv
	.venv/bin/pip install -q -U pip
	.venv/bin/pip install -q numpy pyyaml fastapi "uvicorn[standard]" pytest httpx

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
	@docker run --rm caddy:2.10-alpine caddy hash-password

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
