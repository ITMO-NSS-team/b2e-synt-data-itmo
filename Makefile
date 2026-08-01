PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
export PYTHONPATH := .

OPENAPI ?= Heimdall_openapi.json
DATA    ?= data
SEED    ?= 20260801
N       ?= 300000

.PHONY: help setup catalog data data-small validate stats doc serve test clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/' | sort

setup:  ## окружение и зависимости
	python3 -m venv .venv
	.venv/bin/pip install -q -U pip
	.venv/bin/pip install -q numpy pyyaml fastapi "uvicorn[standard]" pytest

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

test:  ## тесты (корпус собирается внутри тестов)
	$(PY) -m pytest tests -q

clean:
	rm -rf .pytest-data .pytest_cache **/__pycache__
