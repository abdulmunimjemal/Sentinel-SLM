.PHONY: setup install clean check format

setup:
	python -m venv .venv
	@echo "Virtual environment created. Activate with 'source .venv/bin/activate'"

install:
	pip install -r requirements.txt
	pre-commit install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check:
	pre-commit run --all-files

format:
	ruff format .
