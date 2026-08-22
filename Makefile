.PHONY: install test lint verify model

install:
	uv venv --python 3.12 .venv
	uv pip install -e '.[dev,cpu]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff format --check .
	.venv/bin/ruff check .

verify: lint test

model:
	.venv/bin/qc model fetch

