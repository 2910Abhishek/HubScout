# HubScout

Multi-agent ML project architect (LangGraph). Takes an ML task in plain language, clarifies
constraints incl. deployment mode (api / open_weight / compare), researches with parallel scouts
grounded in live registries, checks existence + constraints in code, verifies the top pick in a
sandbox via a Google ADK verifier over A2A, runs a critic loop, outputs a sourced Blueprint.
Production-ready, never deployed.

## Read first
- Spec (source of truth): docs/HubScout_Project_Card.md — read before any design decision.
- Progress: docs/PROGRESS.md — read at session start, update after each task.
- Phase summaries: docs/phases/PHASE_N_README.md. Decisions: docs/decisions/ (ADRs).

## Who does what
- Claude writes, runs, tests and commits ALL code. Never ask the user to run a command to
  verify work — run it and show the result. Sudo-only steps: give minimal exact commands with a
  one-line reason each, wait for confirmation, then verify.
- The user reads docs, asks questions, and tests graphs in LangGraph Studio.
- One phase at a time; each starts in plan mode. After a phase: stop, learning mode, no code
  changes unless asked, until the user writes "start phase N".

## Stack (do not add/remove/swap without asking)
Python 3.12 (uv) · LangGraph + Studio · LangChain (ChatOpenAI only) · OpenRouter free models +
Ollama fallback · Google ADK + a2a-sdk · HF MCP server · FastMCP (arxiv-mcp, ml-insights-mcp) ·
SearXNG · OpenRouter models API + Artificial Analysis API · Postgres + pgvector, BM25, bge-m3,
bge-reranker · LangGraph Postgres checkpointer + Store · Pydantic v2 · hardened Docker sandbox ·
transformers/torch/evaluate · DeepEval + pytest · Langfuse (self-hosted) · FastAPI + SSE ·
Streamlit · Redis · Docker Compose · GitHub Actions · ruff, mypy, pre-commit, pip-audit, bandit,
Trivy, Alembic.

## Commands
- Install: `uv sync`
- Tests (unit): `make test` · integration (live, local only): `make test-integration`
- Lint + types: `make lint` · `make typecheck` · everything: `make check`
- Infra: `make up` (core) · `make up-obs` (+ Langfuse) · `make down` · `make ps`
- Prereqs: `make prereqs`
- Studio: `uv run langgraph dev --no-browser` (API on :2024)
- MCP debug: `uv run fastmcp dev inspector <server.py>`

## Hard rules
- Never invent model IDs, dataset IDs, package names or library APIs. Verify via Context7
  (libraries), HF MCP (models/datasets), OpenRouter catalogue, or PyPI. Docs beat memory.
- Fetch Context7 docs before coding against LangGraph, LangChain, ADK, a2a-sdk, FastMCP,
  DeepEval, Langfuse, OpenRouter or Artificial Analysis.
- Never read, print or commit `.env`. `.env.example` lists every variable, no values.
- All config via pydantic-settings (`app/config.py`). No hardcoded URLs, keys or model names.
- Every LLM output that feeds code is validated by a Pydantic schema.
- `trust_remote_code=False` everywhere.
- Unit/CI tests never call live APIs (fakes + recorded fixtures). Live checks are
  `@pytest.mark.integration`, excluded from CI, run locally by Claude.
- OpenRouter free quota: 20 req/min, 50 req/day. Don't burn it on retries or loops.
- Docker CE (context `default`), not Desktop/Podman. SELinux enforcing: named volumes; bind
  mounts need `:Z`.
- A task is not done until the definition-of-done skill passes. Never claim something works
  without running it.
- Ambiguity → ask one clear question.

## Git
- Branch per phase (`phase-N-<name>`), small Conventional Commits after tests pass.
- Phase end: merge to main with `--no-ff`, tag `phase-N-complete`.
- Always ask before `git push`.
