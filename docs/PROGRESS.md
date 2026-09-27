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

### Done
- [x] 1. `scripts/check_prereqs.sh` written; Python 3.12.12 installed via uv
- [x] 2. git init (main + root commit), remote origin, `.gitignore`, card v0.3
- [x] 3. CLAUDE.md, `.mcp.json`, `.claude/settings.json` + format hook, 9 skills, 2 subagents

### Also done
- [x] Docker CE context `default`, user in docker group; MCP servers context7 + huggingface respond
- [x] 4. uv project (Python 3.12), package skeleton, ruff/mypy/pytest config, pre-commit + detect-secrets
- [x] 5. `app/config.py` (grouped pydantic-settings), slim `.env.example`, `scripts/init_env.sh`,
      `scripts/check_env.py`; `.env.infra` generated

### Waiting on the user
- Replace the old generated `.env` with the slim one and paste API keys
  (OPENROUTER_API_KEY, ARTIFICIAL_ANALYSIS_API_KEY, LANGSMITH_API_KEY, HF_TOKEN, TAVILY_API_KEY).
- Ollama fix: `sudo usermod -d /var/lib/ollama ollama && sudo systemctl restart ollama`
  (the `ollama` user's home points to a non-existent /usr/share/ollama from an old install).

### Next
- [ ] 6. Compose (postgres/pgvector, redis, searxng, langfuse profile) + Makefile
- [ ] 7. `app/llm.py` + Redis rate limiter; pick and pull the Ollama model
- [ ] 8. Integration checks (OpenRouter tool + structured output, Ollama fallback, AA key)
- [ ] 9. Smoke graph + `langgraph.json`; verify with `langgraph dev`
- [ ] 10. CI workflow
- [ ] 11. ADRs, phase README, review, merge, tag

## Known issues
- None yet.
