"""Web search for the method scout: Tavily first, self-hosted SearXNG as the fallback.

Web results are leads only; every URL is verified before it can enter a starter kit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    engine: str


class WebSearch(Protocol):
    async def search(
        self, query: str, max_results: int, include_domains: list[str] | None = None
    ) -> list[WebResult]: ...


class WebSearchClient:
    def __init__(self, settings: Settings) -> None:
        self._s = settings.search
        self._timeout = settings.data.http_timeout_s

    async def _tavily(
        self, query: str, max_results: int, include_domains: list[str] | None = None
    ) -> list[WebResult]:
        if self._s.tavily_api_key is None:
            raise ValueError("TAVILY_API_KEY is not set")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._s.tavily_base_url.rstrip('/')}/search",
                headers={"Authorization": f"Bearer {self._s.tavily_api_key.get_secret_value()}"},
                json={
                    "query": query,
                    "max_results": max_results,
                    "search_depth": self._s.tavily_search_depth,
                    **({"include_domains": include_domains} if include_domains else {}),
                },
            )
            resp.raise_for_status()
        return [
            WebResult(r["title"], r["url"], str(r.get("content", ""))[:400], "tavily")
            for r in resp.json().get("results", [])
            if r.get("url") and r.get("title")
        ]

    async def _searxng(self, query: str, max_results: int) -> list[WebResult]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._s.searxng_url.rstrip('/')}/search", params={"q": query, "format": "json"}
            )
            resp.raise_for_status()
        return [
            WebResult(r["title"], r["url"], str(r.get("content", ""))[:400], "searxng")
            for r in resp.json().get("results", [])[:max_results]
            if r.get("url") and r.get("title")
        ]

    async def search(
        self, query: str, max_results: int, include_domains: list[str] | None = None
    ) -> list[WebResult]:
        if self._s.search_backend == "tavily" and self._s.tavily_api_key is not None:
            try:
                return await self._tavily(query, max_results, include_domains)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.warning("Tavily search failed (%s); falling back to SearXNG", exc)
        if include_domains:  # SearXNG: express the domain filter as site: operators
            query = " ".join([query, *(f"site:{d}" for d in include_domains)])
        try:
            return await self._searxng(query, max_results)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("SearXNG search failed: %s", exc)
            return []
