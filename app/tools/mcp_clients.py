"""MCP clients. The official Hugging Face MCP server gives scouts live Hub search."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import Settings

# Only the tools the open-weight scout needs; fewer tools = more reliable small-model tool use.
HF_SCOUT_TOOLS = frozenset({"hub_repo_search", "hub_repo_details"})


async def load_hf_tools(
    settings: Settings, allowed: frozenset[str] = HF_SCOUT_TOOLS
) -> list[BaseTool]:
    """Connect to the HF MCP server (streamable HTTP) and return the allowed tools.

    Each tool call opens its own short-lived MCP session, so nothing needs closing.
    """
    headers: dict[str, str] = {}
    if settings.data.hf_token is not None:
        headers["Authorization"] = f"Bearer {settings.data.hf_token.get_secret_value()}"
    client = MultiServerMCPClient(
        {
            "huggingface": {
                "transport": "streamable_http",
                "url": settings.data.hf_mcp_url,
                "headers": headers,
                "timeout": settings.data.http_timeout_s,
            }
        }
    )
    tools = await client.get_tools()
    return [tool for tool in tools if tool.name in allowed]
