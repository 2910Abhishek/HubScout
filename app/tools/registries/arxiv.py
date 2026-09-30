"""arXiv API client: search papers and confirm that arXiv ids exist."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from defusedxml import ElementTree

from app.config import Settings

_ATOM = "{http://www.w3.org/2005/Atom}"
_ID = re.compile(r"arxiv\.org/abs/([\w.\-/]+?)(v\d+)?$")
_WORD = re.compile(r"[A-Za-z0-9]+")
RETRY_DELAY_S = 3.0  # arXiv asks for ~3 s between requests
MAX_TERMS = 4  # arXiv ANDs every term; more terms quickly means zero results


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    summary: str
    published: str

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class PaperLookup(Protocol):
    async def search(self, query: str, max_results: int) -> list[Paper]: ...

    async def get(self, arxiv_ids: list[str]) -> list[Paper]: ...


def parse_feed(xml: str) -> list[Paper]:
    """Parse an arXiv Atom feed (defusedxml: safe against XML bombs)."""
    root = ElementTree.fromstring(xml)
    papers: list[Paper] = []
    for entry in root.iter(f"{_ATOM}entry"):
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        match = _ID.search(raw_id)
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        if not match or not title or title == "Error":
            continue
        papers.append(
            Paper(
                arxiv_id=match.group(1),
                title=title,
                summary=" ".join((entry.findtext(f"{_ATOM}summary") or "").split())[:500],
                published=(entry.findtext(f"{_ATOM}published") or "")[:10],
            )
        )
    return papers


class ArxivClient:
    def __init__(self, settings: Settings) -> None:
        self._url = settings.data.arxiv_api_url
        self._timeout = settings.data.http_timeout_s

    async def _query(self, params: dict[str, str]) -> list[Paper]:
        # The arXiv API is often slow or briefly returns 503: one paced retry, longer timeout.
        async with httpx.AsyncClient(timeout=self._timeout * 2, follow_redirects=True) as client:
            for attempt in (1, 2):
                try:
                    resp = await client.get(self._url, params=params)
                    resp.raise_for_status()
                    return parse_feed(resp.text)
                except httpx.HTTPError:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(RETRY_DELAY_S)
        return []

    async def search(self, query: str, max_results: int) -> list[Paper]:
        terms = " AND ".join(f"all:{word}" for word in _WORD.findall(query)[:MAX_TERMS])
        if not terms:
            return []
        return await self._query(
            {"search_query": terms, "max_results": str(max_results), "sortBy": "relevance"}
        )

    async def get(self, arxiv_ids: list[str]) -> list[Paper]:
        if not arxiv_ids:
            return []
        return await self._query(
            {"id_list": ",".join(arxiv_ids), "max_results": str(len(arxiv_ids))}
        )
