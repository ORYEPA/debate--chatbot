SHELL := /bin/sh

.PHONY: help
help: 
	@echo "make help        - lista de comandos"
	@echo "make install     - instala requisitos locales (Python 3.11+, pip)"
	@echo "make test        - ejecuta tests (local)"
	@echo "make run         - levanta docker compose (api + redis + ollama)"
	@echo "make down        - baja servicios"
	@echo "make clean       - baja y borra volúmenes"

.PHONY: install
install:
	@python --version || (echo "❗ Instala Python 3.11+ primero"; exit 1)
	@pip --version || (echo "❗ Instala pip primero"; exit 1)
	@pip install -r fastapi/requirements.txt

.PHONY: test
test:
	@python -m pytest -q fastapi/tests

.PHONY: run
run:
	docker compose up -d --build

.PHONY: down
down:
	docker compose down

.PHONY: clean
clean:
	docker compose down -v --remove-orphans
