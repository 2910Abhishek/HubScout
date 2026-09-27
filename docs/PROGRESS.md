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

### Done
- [x] 1. `scripts/check_prereqs.sh` written; Python 3.12.12 installed via uv
- [x] 2. git init (main + root commit), remote origin, `.gitignore`, card v0.3
- [x] 3. CLAUDE.md, `.mcp.json`, `.claude/settings.json` + format hook, 9 skills, 2 subagents

### Waiting on the user (sudo + re-login)
- A: `sudo usermod -aG docker $USER` → log out/in
- B: `sudo rm /etc/systemd/system/ollama.service && sudo systemctl daemon-reload && sudo systemctl restart ollama`
- Then restart Claude Code. **Resume by**: `docker context use default`, re-run
  `scripts/check_prereqs.sh` until all PASS, and check that the context7 + huggingface MCP servers respond.

### Next
- [ ] 4. uv project, package skeleton, dev tooling, pre-commit (then pipe-test the format hook)
- [ ] 5. `app/config.py` + `.env.example`; user fills `.env`
- [ ] 6. Compose (postgres/pgvector, redis, searxng, langfuse profile) + Makefile
- [ ] 7. `app/llm.py` + Redis rate limiter; pick and pull the Ollama model
- [ ] 8. Integration checks (OpenRouter tool + structured output, Ollama fallback, AA key)
- [ ] 9. Smoke graph + `langgraph.json`; verify with `langgraph dev`
- [ ] 10. CI workflow
- [ ] 11. ADRs, phase README, review, merge, tag

## Known issues
- None yet.
