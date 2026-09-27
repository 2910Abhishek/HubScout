---
name: check-library-docs
description: Fetch current documentation via the Context7 MCP before writing or changing code that uses LangGraph, LangChain, Google ADK, a2a-sdk, FastMCP, DeepEval, Langfuse, pydantic-settings, or the OpenRouter / Artificial Analysis APIs. Use whenever you are unsure of an import path, signature, or behaviour.
---

# Check library docs

1. Find the installed version: `uv pip show <package> | grep Version` (or `uv.lock`).
2. Context7: `resolve-library-id` with the library name → pick the official ID (prefer ones matching our version).
3. `get-library-docs` / `query-docs` with a focused topic (e.g. "interrupt", "with_fallbacks", "PostgresSaver setup").
4. For HTTP APIs not in Context7 (OpenRouter, Artificial Analysis): fetch the official docs page.
5. Before coding, write down: import path, signature, one gotcha. If docs contradict memory, **docs win** — mention the difference to the user.
6. Confirm by running: a tiny `uv run python -c "from X import Y; help(Y)"` or a unit test.

Never guess an API. If docs are unavailable, say so and ask.
