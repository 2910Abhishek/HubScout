"""Test doubles for graph nodes: scripted LLMs, an in-memory Hub, and a fake MCP tool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from app.config import Settings, load_settings
from app.graph.deps import Deps
from app.tools.registries.hub import ModelFacts


@dataclass
class ScriptedLlm:
    """Returns queued objects per schema (structured output) or AIMessages (tool mode)."""

    structured: dict[type[BaseModel], list[Any]] = field(default_factory=dict)
    tool_replies: list[AIMessage] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def __call__(
        self,
        tier: str,
        *,
        tools: list[BaseTool] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> Runnable[Any, Any]:
        def respond(_: Any) -> Any:
            if schema is not None:
                self.calls.append(f"{tier}:{schema.__name__}")
                return self.structured[schema].pop(0)
            self.calls.append(f"{tier}:tools")
            return self.tool_replies.pop(0) if self.tool_replies else AIMessage(content="done")

        return RunnableLambda(respond)


class FakeHub:
    def __init__(self, facts: dict[str, ModelFacts]) -> None:
        self.facts = facts

    async def model_facts(self, repo_id: str) -> ModelFacts:
        return self.facts.get(
            repo_id, ModelFacts(repo_id=repo_id, exists=False, retrieved_at=datetime.now(UTC))
        )


def facts(repo_id: str, **kw: Any) -> ModelFacts:
    base: dict[str, Any] = {
        "exists": True,
        "retrieved_at": datetime(2026, 9, 27, tzinfo=UTC),
        "licence": "apache-2.0",
        "pipeline_tag": "automatic-speech-recognition",
        "params": 240_000_000,
        "languages": ["en", "hi"],
        "downloads": 1_000_000,
    }
    return ModelFacts(repo_id=repo_id, **{**base, **kw})


def make_search_tool(record: list[dict[str, Any]]) -> BaseTool:
    @tool
    def hub_repo_search(query: str, filters: list[str] | None = None) -> str:
        """Search Hugging Face repositories."""
        record.append({"query": query, "filters": filters})
        return "openai/whisper-small (2.9M downloads)\nfake/huge-model\nnobody/does-not-exist"

    return hub_repo_search


def make_deps(
    llm: ScriptedLlm, hub: FakeHub, tools: list[BaseTool], settings: Settings | None = None
) -> Deps:
    async def load_tools() -> list[BaseTool]:
        return tools

    return Deps(
        settings=settings or load_settings(env_files=None), llm=llm, hub=hub, load_tools=load_tools
    )


Factory = Callable[..., Runnable[Any, Any]]
