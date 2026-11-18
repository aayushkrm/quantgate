.PHONY: setup lint fmt type test demo readme results report check-results site release

# Targets run the venv's tools through `uv run`. Without this, uv resyncs the venv to the extras
# it was asked for, and removes llama-cpp-python, which the eval and perf runs need.
export UV_NO_SYNC := 1

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

# Rewrites the generated blocks of README.md from results/report.json (or the no-results text).
readme:
	uv run python scripts/render_readme.py

# The full pre-registered run: hours on the measurement laptop. Runs the quality suites and then
# perf, one variant at a time. Resumable. Read docs/prereg.md before running it.
results:
	uv run quantgate eval -c configs/variants.yaml --out results

# Markdown, Pareto plots and the static page from committed results. Fails clearly without them.
report:
	uv run quantgate report results --out-dir site

# README blocks must match the committed results, the README must have no banned words, and the
# gate must agree with the stored decisions. Uses a SYNTHETIC fixture until results are committed.
check-results:
	uv run python scripts/render_readme.py --check
	uv run python scripts/check_results.py

# Static site from committed results. With no results it prints why and exits 0, so CI stays
# green before the first run.
site:
	@if [ -f results/report.json ]; then \
		uv run quantgate report results --out-dir site; \
	else \
		echo "site: no results/report.json committed; nothing to build"; \
	fi

# Source and wheel in dist/, then a check of the wheel contents.
release:
	rm -rf dist
	uv build
	uv run python scripts/check_wheel.py
