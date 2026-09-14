.PHONY: install test lint typecheck verify

install:
	uv venv --python 3.12 .venv
	uv pip install -e '.[dev,cpu]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff format --check .
	.venv/bin/ruff check .

typecheck:
	.venv/bin/mypy

verify: lint typecheck test
