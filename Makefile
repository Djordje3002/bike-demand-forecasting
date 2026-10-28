.PHONY: install train test check
install:
	python -m pip install -e ".[dev]"
train:
	python -m bike_forecast train
test:
	python -m pytest -q
check:
	python -m ruff check .
	python -m ruff format --check .
