# HubScout progress

## Current: Phase 1 — Foundation (branch `phase-1-foundation`)

Approved plan: `~/.claude/plans/noble-floating-thacker.md` (copy of key decisions below).

### Decisions (2026-09-27)
- The prompt's stack beats card v0.2; card updated to v0.3 in `docs/HubScout_Project_Card.md`.
- Docker CE (context `default`), not Docker Desktop. User joins the `docker` group.
- Ollama: Fedora RPM 0.12.11, CPU only (the RPM has no CUDA); small (≤4B) tool-calling model.
- OpenRouter: no purchased credits → 20 req/min **and 50 req/day** on `:free` models.
- Postgres host port 5433 (5432 is taken on the host).
- The card's "cost & deployment analyst" is assumed to land in Phase 3.
- `.env` holds only external API keys; non-secret defaults live in `app/config.py`;
  generated local infra secrets live in `.env.infra` (user request).
- No AI co-author trailers in commits (portfolio project).
- Web search: Tavily primary, SearXNG fallback. Semantic Scholar dropped (keyless API returns
  429 constantly; keys need an institutional affiliation). Card v0.4.
- Ollama model: `qwen3:4b-instruct` (no thinking tokens; fast enough on CPU).
- 2026-09-29: scope reduced to a verified ML starter kit (README with <=15 links across
  models, datasets, methods). Project card rewritten as v1.0; see its roadmap for phases.

### Done
- [x] 1. `scripts/check_prereqs.sh` written; Python 3.12.12 installed via uv
- [x] 2. git init (main + root commit), remote origin, `.gitignore`, card v0.3
- [x] 3. CLAUDE.md, `.mcp.json`, `.claude/settings.json` + format hook, 9 skills, 2 subagents

### Also done
- [x] Docker CE context `default`, user in docker group; MCP servers context7 + huggingface respond
- [x] 4. uv project (Python 3.12), package skeleton, ruff/mypy/pytest config, pre-commit + detect-secrets
- [x] 5. `app/config.py` (grouped pydantic-settings), slim `.env.example`, `scripts/init_env.sh`,
      `scripts/check_env.py`; `.env.infra` generated

### Fast-track POC (user request, 2026-09-27)
The user asked to skip ahead to a working agent for hands-on learning, so Phase 1's remaining
steps and a Phase 2 thin slice were combined:
- [x] 6. Docker Compose (postgres+pgvector, redis, searxng; Langfuse under `observability`) + Makefile
- [x] 7. `app/llm.py` + Redis rate limiter; Ollama `qwen3:4b-instruct` pulled
- [x] Schemas, Hub facts client, code-only checks, HF MCP client
- [x] Graph "hubscout": clarify -> ask_user -> plan -> review_plan -> scout -> check -> aggregate
- [x] 10. CI workflow
- [x] Live end-to-end run verified through `langgraph dev` (see docs/phases/POC_README.md)

### Next
- [ ] 8. Integration tests (`tests/integration/`) for OpenRouter, Ollama fallback, Artificial
      Analysis, HF MCP (live checks were run by hand so far)
- [ ] 11. ADRs (OpenRouter+Ollama, Store vs Mem0, Docker sandbox, Redis, Tavily)
- [ ] code-reviewer pass, merge to main, tag
- [ ] Card v1.0 Phase 2 (starter kit): dataset scout + checks + previews, method scout (arXiv,
      HF Papers, Tavily), link verifier, StarterKit schema, README renderer, parallel scouts

## Known issues
- Free OpenRouter models are often throttled upstream (HTTP 429); on 2026-09-27 only Nemotron
  answered, so both strong and cheap tiers use it. Fallback to Ollama works but is slow on CPU.
- LangGraph warns that Pydantic types in checkpoints are "unregistered"; harmless now, to be
  addressed with an explicit serializer allowlist when the Postgres checkpointer is added.
- API and compare modes are not implemented yet (the blueprint says so).
