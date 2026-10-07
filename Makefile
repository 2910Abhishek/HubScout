# HubScout developer commands. Run `make help` for the list.

# A globally activated venv (e.g. pyenv) would make uv warn and ignore it; drop it here.
unexport VIRTUAL_ENV

COMPOSE := docker compose --env-file .env.infra
UV := uv run

.DEFAULT_GOAL := help
.PHONY: help env prereqs up up-obs down down-obs logs ps test test-integration lint format typecheck audit check studio

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env: ## Create .env (API keys) and .env.infra (generated secrets) if missing
	@scripts/init_env.sh

.env.infra:
	@scripts/init_env.sh

prereqs: ## Check system prerequisites
	@scripts/check_prereqs.sh

up: .env.infra ## Start core services (postgres, redis, searxng) and wait until healthy
	$(COMPOSE) up -d --wait --wait-timeout 180

up-obs: .env.infra ## Start core services plus the Langfuse stack
	$(COMPOSE) --profile observability up -d --wait --wait-timeout 420

down: ## Stop core services (data volumes are kept)
	$(COMPOSE) down

down-obs: ## Stop everything including Langfuse (data volumes are kept)
	$(COMPOSE) --profile observability down

logs: ## Follow logs (SERVICE=name to filter)
	$(COMPOSE) --profile observability logs -f --tail=100 $(SERVICE)

ps: ## Show service status
	$(COMPOSE) --profile observability ps

test: ## Unit tests (no network)
	$(UV) pytest

test-integration: ## Live checks against real services (local only; uses OpenRouter quota)
	$(UV) pytest -m integration -v

lint: ## Lint and check formatting
	$(UV) ruff check .
	$(UV) ruff format --check .

format: ## Autofix lint issues and format
	$(UV) ruff check --fix .
	$(UV) ruff format .

typecheck: ## Strict mypy on app/
	$(UV) mypy app

audit: ## Dependency vulnerabilities and security lint
	$(UV) pip-audit --skip-editable
	$(UV) bandit -q -c pyproject.toml -r app mcp_servers verifier

check: lint typecheck test audit ## Everything CI runs

studio: ## Start the LangGraph dev server for Studio (API on :2024)
	$(UV) langgraph dev --no-browser
