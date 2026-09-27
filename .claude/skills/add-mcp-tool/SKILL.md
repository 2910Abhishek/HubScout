---
name: add-mcp-tool
description: Recipe for adding a tool to HubScout's FastMCP servers (arxiv-mcp, ml-insights-mcp) - typed signature, docstring, in-memory client test, and manifest snapshot via fastmcp inspect. Use whenever creating or changing an MCP tool.
---

# Add an MCP tool

1. **Docs**: check-library-docs for FastMCP (decorators, Context, in-memory Client). Verify against the installed version.
2. **Typed tool**: `@mcp.tool` function with fully typed params (use `Annotated[..., Field(description=...)]`) and a Pydantic return model. No `Any`. Validate inputs; raise clear errors.
3. **Docstring**: the first line says what it does and when to use it (the LLM reads it). Document units (GB, USD, tokens).
4. **Pure core**: put logic in a plain function the tool calls, so it is unit-testable without MCP.
5. **In-memory test**: `async with Client(mcp) as c: await c.call_tool("name", {...})` in `tests/unit/mcp/`. External HTTP is faked (respx / recorded fixtures) — never live.
6. **Manifest snapshot**: `uv run fastmcp inspect <server.py> --format mcp` (check the current CLI flags via `--help`) → write it to `mcp_servers/<server>/manifest.snapshot.json`; a test compares the live manifest to the snapshot.
7. Run definition-of-done, then git-commit (`feat(mcp): add <tool> to <server>`).
