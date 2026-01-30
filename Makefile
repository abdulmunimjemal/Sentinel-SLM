.PHONY: setup install clean check format demo demo-build demo-up demo-down demo-logs

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

# -----------------------------------------------------------------------------
# Demo Targets
# -----------------------------------------------------------------------------
demo-build:  ## Build Docker images for the demo
	docker-compose -f demo/docker-compose.yml build

demo-up:  ## Start the demo (API + UI)
	docker-compose -f demo/docker-compose.yml up -d
	@echo "✅ Demo starting..."
	@echo "   API:       http://localhost:8000"
	@echo "   Dashboard: http://localhost:8501"

demo-down:  ## Stop the demo
	docker-compose -f demo/docker-compose.yml down

demo-logs:  ## View logs
	docker-compose -f demo/docker-compose.yml logs -f

demo: demo-build demo-up  ## Build and run the demo
