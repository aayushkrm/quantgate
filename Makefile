.PHONY: setup lint fmt type test demo report check-results site

setup:
	uv venv -p 3.11 .venv
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

type:
	uv run mypy src

test:
	uv run pytest -m "not model"

demo:
	uv run quantgate demo

report:
	uv run quantgate report

check-results:
	@echo "check-results: not implemented yet (needs the README render check)" && exit 1

site:
	@echo "site: not implemented yet (needs quantgate report)" && exit 1
