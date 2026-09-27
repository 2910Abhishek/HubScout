# HubScout

Multi-agent ML project architect. Describe an ML task in plain language; HubScout clarifies your
constraints, researches the Hugging Face Hub through the official HF MCP server, verifies every
candidate **in code** (existence, licence, VRAM, task, language), and returns a sourced
blueprint. Built with LangGraph, OpenRouter free models (Ollama fallback), and Pydantic.

Full specification: [docs/HubScout_Project_Card.md](docs/HubScout_Project_Card.md) ·
Progress: [docs/PROGRESS.md](docs/PROGRESS.md)

## Status
- POC, open-weight path end to end: [docs/phases/POC_README.md](docs/phases/POC_README.md)

## Project structure
```
app/
  config.py              all settings (defaults; secrets from .env / .env.infra)
  llm.py                 tiered models: OpenRouter + Ollama fallback
  rate_limit.py          Redis limiter for OpenRouter's free quota (20/min, 50/day)
  schemas/               Constraints, ResearchPlan, candidates, Blueprint (Pydantic)
  graph/
    build.py             the "hubscout" graph (registered in langgraph.json)
    state.py, deps.py    graph state and injected dependencies
    prompts.py           system prompts
    nodes/               clarifier, planner, scout, checker, aggregator
  tools/
    mcp_clients.py       Hugging Face MCP client (langchain-mcp-adapters)
    registries/hub.py    facts from the Hub API (source of truth)
    checks/              code-only candidate checks
  retrieval/ memory/ guardrails/ api/      later phases
mcp_servers/             custom FastMCP servers (arxiv-mcp, ml-insights-mcp; later)
verifier/                Google ADK verifier over A2A (later)
ui/ evals/               Streamlit UI and evaluation suite (later)
tests/unit/              offline tests (fakes only); tests/integration/ for live checks
infra/                   SearXNG settings, Postgres init
scripts/                 prerequisite, env init and env check scripts
docs/                    project card, progress, phase READMEs, ADRs
```

## Stack
Python 3.12 (uv) · LangGraph + Studio · LangChain (ChatOpenAI) · OpenRouter + Ollama ·
Hugging Face MCP · Pydantic v2 · Redis · Postgres + pgvector · SearXNG / Tavily · Langfuse ·
Docker Compose · GitHub Actions · ruff, mypy, pytest, pre-commit, pip-audit, bandit.
