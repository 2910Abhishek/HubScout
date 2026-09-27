"""Dependencies injected into graph nodes (real ones in production, fakes in tests)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.config import LlmTier, Settings, get_settings
from app.llm import get_llm
from app.tools.mcp_clients import load_hf_tools
from app.tools.registries.hub import HubClient, HubLookup


class LlmFactory(Protocol):
    def __call__(
        self,
        tier: LlmTier,
        *,
        tools: list[BaseTool] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> Runnable[LanguageModelInput, Any]: ...


@dataclass(frozen=True)
class Deps:
    settings: Settings
    llm: LlmFactory
    hub: HubLookup
    load_tools: Callable[[], Awaitable[list[BaseTool]]]


def default_deps() -> Deps:
    settings = get_settings()

    def llm(
        tier: LlmTier,
        *,
        tools: list[BaseTool] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> Runnable[LanguageModelInput, Any]:
        return get_llm(tier, tools=tools, schema=schema, settings=settings)

    async def tools() -> list[BaseTool]:
        return await load_hf_tools(settings)

    return Deps(settings=settings, llm=llm, hub=HubClient(settings), load_tools=tools)
