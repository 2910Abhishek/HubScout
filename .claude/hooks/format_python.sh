#!/usr/bin/env bash
# PostToolUse hook: format + autofix Python files after Claude edits them.
# Reads the hook JSON on stdin; ignores non-.py files. Never blocks the edit.
set -uo pipefail
f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
[[ -n "$f" && "$f" == *.py && -f "$f" ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0
uv run --quiet ruff format "$f" >/dev/null 2>&1
uv run --quiet ruff check --fix --quiet "$f" >/dev/null 2>&1
exit 0
