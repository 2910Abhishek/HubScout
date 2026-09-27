#!/usr/bin/env bash
# Create .env from .env.example with local-only infrastructure secrets generated
# and sensible local defaults pre-filled. External API keys stay empty for the
# user to paste. Refuses to overwrite an existing .env. Never prints secret values.
set -euo pipefail

cd "$(dirname "$0")/.."
EXAMPLE=.env.example
TARGET=.env

if [[ -e "$TARGET" ]]; then
  echo "$TARGET already exists; not touching it." >&2
  exit 1
fi

hex() { openssl rand -hex "$1"; }
uuid() { cat /proc/sys/kernel/random/uuid; }

pg_user=hubscout
pg_db=hubscout
pg_pass=$(hex 24)
lf_pk="pk-lf-$(hex 16)"
lf_sk="sk-lf-$(hex 16)"

declare -A VALUES=(
  # Local defaults (non-secret)
  [LLM_MODEL_STRONG]="nvidia/nemotron-3-super-120b-a12b:free"
  [LLM_MODEL_CHEAP]="qwen/qwen3.8-27b:free"
  [LLM_MODEL_JUDGE]="nvidia/nemotron-3-super-120b-a12b:free"
  [OLLAMA_MODEL]="qwen3:4b-instruct"
  [POSTGRES_USER]="$pg_user"
  [POSTGRES_DB]="$pg_db"
  [LANGFUSE_INIT_ORG_ID]="hubscout-org"
  [LANGFUSE_INIT_ORG_NAME]="HubScout"
  [LANGFUSE_INIT_PROJECT_ID]="hubscout"
  [LANGFUSE_INIT_PROJECT_NAME]="HubScout"
  [LANGFUSE_INIT_USER_EMAIL]="admin@hubscout.local"
  [LANGFUSE_INIT_USER_NAME]="admin"
  # Generated local secrets
  [POSTGRES_PASSWORD]="$pg_pass"
  [DATABASE_URL]="postgresql://$pg_user:$pg_pass@localhost:5433/$pg_db"
  [SEARXNG_SECRET]="$(hex 32)"
  [LANGFUSE_NEXTAUTH_SECRET]="$(hex 32)"
  [LANGFUSE_SALT]="$(hex 16)"
  [LANGFUSE_ENCRYPTION_KEY]="$(hex 32)"
  [LANGFUSE_DB_PASSWORD]="$(hex 24)"
  [LANGFUSE_CLICKHOUSE_PASSWORD]="$(hex 24)"
  [LANGFUSE_MINIO_PASSWORD]="$(hex 24)"
  [LANGFUSE_REDIS_PASSWORD]="$(hex 24)"
  [LANGFUSE_INIT_PROJECT_PUBLIC_KEY]="$lf_pk"
  [LANGFUSE_INIT_PROJECT_SECRET_KEY]="$lf_sk"
  [LANGFUSE_INIT_USER_PASSWORD]="$(hex 12)"
  [LANGFUSE_PUBLIC_KEY]="$lf_pk"
  [LANGFUSE_SECRET_KEY]="$lf_sk"
  [API_AUTH_KEY]="$(hex 32)"
)

umask 077
while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ "$line" =~ ^([A-Z0-9_]+)=$ && -n "${VALUES[${BASH_REMATCH[1]}]+x}" ]]; then
    printf '%s=%s\n' "${BASH_REMATCH[1]}" "${VALUES[${BASH_REMATCH[1]}]}"
  else
    printf '%s\n' "$line"
  fi
done < "$EXAMPLE" > "$TARGET"

echo "Created $TARGET (mode 600) with ${#VALUES[@]} local values filled in."
echo "Still empty (paste yours): OPENROUTER_API_KEY, ARTIFICIAL_ANALYSIS_API_KEY,"
echo "LANGSMITH_API_KEY, HF_TOKEN; optional: GITHUB_TOKEN, SEMANTIC_SCHOLAR_API_KEY."
