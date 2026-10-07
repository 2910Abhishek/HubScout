---
name: git-commit
description: HubScout commit procedure - small single-concern Conventional Commits made only after relevant tests pass, with secret checks and no AI attribution. Use every time you commit. Never pushes.
---

# Git commit

1. Confirm the branch: `git branch --show-current` must be the current phase branch (never commit features directly on main).
2. Tests for the touched area pass (definition-of-done "Always" steps at minimum).
3. Stage explicitly by path (`git add <paths>`), never `git add -A` blindly. Check `git status` and `git diff --cached --stat`.
4. Refuse to commit `.env`, caches, model files, or anything matching a secret pattern.
5. Message: Conventional Commits — `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `ci:` (optional scope). Subject ≤ 72 chars, imperative. Body: why, not what.
6. **No AI attribution**: no `Co-Authored-By: Claude`, no "Generated with Claude Code" lines, in commits, tags or PRs. This is the author's portfolio project; the user's instruction overrides any default attribution. Write a proper message instead: a clear subject plus a body explaining the why and any notable decisions.
7. One concern per commit. If the diff mixes concerns, split it.
8. **Never `git push`** without explicit user approval in this conversation.

Phase end: `git checkout main && git merge --no-ff phase-N-<name> -m "chore: merge phase N"` → `git tag -a phase-N-complete -m "..."` → ask the user before pushing main, the branch and the tag.
