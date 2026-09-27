#!/usr/bin/env bash
# Verify every system prerequisite for HubScout. Exits non-zero if any check fails.
set -uo pipefail

FAILS=0
MIN_DISK_GB="${MIN_DISK_GB:-20}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

pass() { printf '  \033[32mPASS\033[0m  %-28s %s\n' "$1" "$2"; }
fail() { printf '  \033[31mFAIL\033[0m  %-28s %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
info() { printf '  \033[34mINFO\033[0m  %-28s %s\n' "$1" "$2"; }

have() { command -v "$1" >/dev/null 2>&1; }

echo "HubScout prerequisite check"

# git
if have git; then pass "git" "$(git --version)"; else fail "git" "not installed"; fi

# uv + Python 3.12 managed by uv
if have uv; then
  pass "uv" "$(uv --version)"
  if py=$(uv python find 3.12 2>/dev/null); then
    pass "python 3.12 (uv)" "$("$py" --version) at $py"
  else
    fail "python 3.12 (uv)" "run: uv python install 3.12"
  fi
else
  fail "uv" "not installed (curl -LsSf https://astral.sh/uv/install.sh | sh)"
  fail "python 3.12 (uv)" "needs uv"
fi

# Docker CE (not Docker Desktop, not Podman)
if have docker; then
  ctx=$(docker context show 2>/dev/null || echo "?")
  if server_os=$(docker info --format '{{.OperatingSystem}}' 2>/dev/null); then
    if [[ "$server_os" == *"Docker Desktop"* ]]; then
      fail "docker engine" "context '$ctx' is Docker Desktop; run: docker context use default"
    else
      pass "docker engine" "$(docker version --format '{{.Server.Version}}') on '$server_os' (context $ctx)"
    fi
  else
    fail "docker engine" "cannot reach daemon via context '$ctx' (docker group? daemon running?)"
  fi
  if cv=$(docker compose version --short 2>/dev/null); then
    pass "docker compose plugin" "v$cv"
  else
    fail "docker compose plugin" "not installed"
  fi
else
  fail "docker engine" "docker CLI not installed"
fi

# Current user in docker group (checks the live session, not just /etc/group)
if id -nG | tr ' ' '\n' | grep -qx docker; then
  pass "docker group" "$USER is a member"
elif getent group docker | grep -qw "$USER"; then
  fail "docker group" "added but session not refreshed; log out and back in"
else
  fail "docker group" "$USER not a member (sudo usermod -aG docker \$USER)"
fi

# Node.js >= 18 and npx
if have node; then
  major=$(node -p 'process.versions.node.split(".")[0]')
  if (( major >= 18 )); then pass "node >= 18" "$(node --version)"; else fail "node >= 18" "found $(node --version)"; fi
else
  fail "node >= 18" "not installed"
fi
if have npx; then pass "npx" "$(npx --version)"; else fail "npx" "not installed"; fi

# jq
if have jq; then pass "jq" "$(jq --version)"; else fail "jq" "not installed"; fi

# Ollama installed and serving
if have ollama; then
  if curl -fsS -m 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
    pass "ollama" "serving $(curl -fsS -m 3 "$OLLAMA_URL/api/version" | jq -r .version) at $OLLAMA_URL"
  else
    fail "ollama" "installed but not reachable at $OLLAMA_URL"
  fi
else
  fail "ollama" "not installed"
fi

# Free disk space in the project directory
free_gb=$(df -BG --output=avail "$(dirname "$0")/.." | tail -1 | tr -dc '0-9')
if (( free_gb >= MIN_DISK_GB )); then
  pass "disk space" "${free_gb} GB free (need ${MIN_DISK_GB})"
else
  fail "disk space" "${free_gb} GB free (need ${MIN_DISK_GB})"
fi

# SELinux (informational)
if have getenforce; then info "selinux" "$(getenforce) (use named volumes; bind mounts need :Z)"; fi

echo
if (( FAILS == 0 )); then
  echo "All prerequisites satisfied."
else
  echo "$FAILS check(s) failed."
fi
exit $(( FAILS > 0 ))
