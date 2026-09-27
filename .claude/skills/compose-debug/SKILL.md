---
name: compose-debug
description: Fixed troubleshooting checklist for an unhealthy or failing Docker Compose service on this Fedora + SELinux + Docker CE machine. Use when docker compose up --wait fails or a service is unhealthy or restarting.
---

# Compose debug checklist

1. Context: `docker context show` must be `default` (Docker CE), not `desktop-linux`.
2. Status: `docker compose ps -a` → which service, what state, what exit code.
3. Logs: `docker compose logs --tail=100 <service>`.
4. Healthcheck: `docker inspect --format '{{json .State.Health}}' <container> | jq` → run the same test command via `docker compose exec <service> <cmd>`.
5. SELinux: bind mounts need `:Z` (or `:z` when shared). Check for denials with `journalctl -t setroubleshoot --since -10min` (read-only; `ausearch` needs sudo, so ask the user if you need it).
6. Volumes/permissions: named volumes preferred; for "permission denied" inside a container, check the image's UID vs the volume owner (`docker compose exec <svc> id`).
7. Ports: `ss -ltn | grep <port>` for host conflicts (host Postgres already uses 5432, so ours is on 5433).
8. Env: `docker compose config` renders the resolved config (it prints secret values, so never paste its output; check with grep for specific keys).
9. Resources: `docker stats --no-stream` (Langfuse + ClickHouse are memory-heavy).
10. Reset one service: `docker compose up -d --force-recreate <service>`. Only delete volumes with user approval.
