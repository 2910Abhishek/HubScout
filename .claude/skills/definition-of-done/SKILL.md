---
name: definition-of-done
description: Run before declaring ANY task or phase complete in HubScout. Executes lint, format check, types, unit tests, relevant integration checks, Compose health and Studio load, and reports real output. Use whenever you are about to say "done", "works", or commit a feature.
---

# Definition of done

Never say something works unless you ran it in this session. Run every step and show the output.

## Always (every task)
1. `uv run ruff check .` → zero errors
2. `uv run ruff format --check .` → no changes needed
3. `uv run mypy app` → `Success`
4. `uv run pytest -q` → all pass (integration tests are excluded by default)
5. `git status` → no `.env`, caches, or stray files staged
6. No secrets in the diff: `git diff --cached | grep -iE '(sk-or-|hf_[A-Za-z0-9]{10}|api[_-]?key\s*=\s*\S+)'` → no hits

## When the task touches these areas
- LLM / external APIs → run the matching `uv run pytest -m integration -k <name>` locally (watch the 50/day OpenRouter budget; one run is enough)
- Compose / infra → `docker compose ps` shows every service `healthy`
- Graphs / nodes → start `uv run langgraph dev --no-browser` in the background, `curl -s localhost:2024/assistants/search -H 'content-type: application/json' -d '{}'` lists the graph, then stop the server
- MCP servers → in-memory client tests + `fastmcp inspect` manifest matches the committed snapshot
- Dependencies → `uv run pip-audit` clean, `uv lock --check` passes
- Security-sensitive code → `uv run bandit -q -r app mcp_servers verifier`

## Phase end (additionally)
- `make check` green; code-reviewer subagent run and important findings fixed
- docs/PROGRESS.md, docs/phases/PHASE_N_README.md, root README updated

## Report format
List each step with PASS/FAIL and the key output line. If anything was skipped, say which and why.
