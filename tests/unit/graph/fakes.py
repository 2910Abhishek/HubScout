"""Test doubles for graph nodes: scripted LLMs, in-memory Hub/arXiv/web, and a fake MCP tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from app.config import Settings, load_settings
from app.graph.deps import Deps
from app.tools.checks.links import LinkStatus
from app.tools.registries.arxiv import Paper
from app.tools.registries.hub import DatasetFacts, ModelFacts
from app.tools.search import WebResult

WHEN = datetime(2026, 9, 27, tzinfo=UTC)


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
                queue = self.structured.get(schema)
                if not queue:
                    raise RuntimeError(f"no scripted {schema.__name__}")
                return queue.pop(0)
            self.calls.append(f"{tier}:tools")
            return self.tool_replies.pop(0) if self.tool_replies else AIMessage(content="done")

        return RunnableLambda(respond)


def facts(repo_id: str, **kw: Any) -> ModelFacts:
    base: dict[str, Any] = {
        "exists": True,
        "retrieved_at": WHEN,
        "licence": "apache-2.0",
        "pipeline_tag": "automatic-speech-recognition",
        "params": 240_000_000,
        "languages": ["en", "hi"],
        "downloads": 1_000_000,
    }
    return ModelFacts(repo_id=repo_id, **{**base, **kw})


def dataset_facts(repo_id: str, **kw: Any) -> DatasetFacts:
    base: dict[str, Any] = {
        "exists": True,
        "retrieved_at": WHEN,
        "licence": "cc-by-4.0",
        "languages": ["hi", "en"],
        "task_categories": ["automatic-speech-recognition"],
        "downloads": 50_000,
        "viewer_ok": True,
        "num_rows": 12_000,
        "splits": ["train", "test"],
        "features": ["audio (Audio)", "sentence (Value)"],
        "sample_rows": [{"audio": "<audio>", "sentence": "namaste duniya"}],
    }
    return DatasetFacts(repo_id=repo_id, **{**base, **kw})


class FakeHub:
    def __init__(
        self,
        models: dict[str, ModelFacts] | None = None,
        datasets: dict[str, DatasetFacts] | None = None,
    ) -> None:
        self.models = models or {}
        self.datasets = datasets or {}

    async def model_facts(self, repo_id: str) -> ModelFacts:
        return self.models.get(
            repo_id, ModelFacts(repo_id=repo_id, exists=False, retrieved_at=WHEN)
        )

    async def dataset_facts(self, repo_id: str) -> DatasetFacts:
        return self.datasets.get(
            repo_id, DatasetFacts(repo_id=repo_id, exists=False, retrieved_at=WHEN)
        )


class FakeArxiv:
    """`search_results` answer searches; `papers` answer id lookups (what exists on arXiv)."""

    def __init__(
        self, papers: list[Paper] | None = None, search_results: list[Paper] | None = None
    ) -> None:
        self.papers = {p.arxiv_id: p for p in [*(papers or []), *(search_results or [])]}
        self.search_results = search_results or []
        self.searches: list[str] = []
        self.lookups: list[list[str]] = []

    async def search(self, query: str, max_results: int) -> list[Paper]:
        self.searches.append(query)
        return self.search_results[:max_results]

    async def get(self, arxiv_ids: list[str]) -> list[Paper]:
        self.lookups.append(list(arxiv_ids))
        return [self.papers[i] for i in arxiv_ids if i in self.papers]


class FakeWeb:
    def __init__(self, results: list[WebResult] | None = None) -> None:
        self.results = results or []

    async def search(self, query: str, max_results: int) -> list[WebResult]:
        return self.results[:max_results]


class FakeLinks:
    def __init__(self, dead: set[str] | None = None) -> None:
        self.dead = dead or set()

    async def check(self, url: str) -> LinkStatus:
        if url in self.dead:
            return LinkStatus(url, False, 404, "HTTP 404")
        return LinkStatus(url, True, 200, "ok")


def make_search_tool(
    record: list[dict[str, Any]], models: str = "", datasets: str = ""
) -> BaseTool:
    @tool
    def hub_repo_search(
        query: str = "",
        filters: list[str] | None = None,
        repo_types: list[str] | None = None,
        sort: str = "downloads",
        limit: int = 20,
    ) -> str:
        """Search Hugging Face repositories."""
        record.append({"query": query, "filters": filters, "repo_types": repo_types})
        if repo_types == ["dataset"]:
            return datasets or "No repositories found for the given criteria."
        return models or "No repositories found for the given criteria."

    return hub_repo_search


def make_deps(
    llm: Any,
    hub: Any,
    tools: list[BaseTool],
    settings: Settings | None = None,
    arxiv: FakeArxiv | None = None,
    web: FakeWeb | None = None,
    links: FakeLinks | None = None,
) -> Deps:
    async def load_tools() -> list[BaseTool]:
        return tools

    cfg = settings or load_settings(env_files=None)
    # Tests never write files into the repository.
    cfg = cfg.model_copy(update={"policy": cfg.policy.model_copy(update={"output_dir": ""})})
    return Deps(
        settings=cfg,
        llm=llm,
        hub=hub,
        load_tools=load_tools,
        arxiv=arxiv or FakeArxiv(),
        web=web or FakeWeb(),
        links=links or FakeLinks(),
    )
