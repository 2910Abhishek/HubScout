"""Method scout: papers (arXiv) and guides/repos (Tavily, SearXNG fallback). Code only.

Results are unverified leads here. The verify node checks every link, and the write node lets
the LLM choose among verified items only.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Literal

from app.graph.deps import Deps
from app.graph.state import HubScoutState
from app.schemas import MethodCandidate

logger = logging.getLogger(__name__)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def classify_web_url(url: str) -> tuple[Literal["paper", "guide", "repo"], str | None]:
    """(kind, arxiv_id) for a web result."""
    if match := _ARXIV_URL.search(url):
        return "paper", match.group(1)
    if "github.com/" in url or "gitlab.com/" in url:
        return "repo", None
    return "guide", None


def make_method_scout_node(deps: Deps) -> Any:
    settings = deps.settings

    async def method_scout(state: HubScoutState) -> dict[str, Any]:
        plan = state["plan"]
        errors: list[str] = []

        async def papers(query: str) -> list[MethodCandidate]:
            try:
                found = await deps.arxiv.search(query, settings.search.arxiv_max_results)
            except Exception as exc:
                errors.append(f"arXiv search failed for '{query}' ({type(exc).__name__})")
                return []
            return [
                MethodCandidate(
                    title=p.title,
                    url=p.url,
                    kind="paper",
                    source="arxiv",
                    snippet=p.summary[:300],
                    arxiv_id=p.arxiv_id,
                    published=p.published,
                )
                for p in found
            ]

        async def web(query: str) -> list[MethodCandidate]:
            results = await deps.web.search(query, settings.search.tavily_max_results)
            out = []
            for r in results:
                kind, arxiv_id = classify_web_url(r.url)
                out.append(
                    MethodCandidate(
                        title=r.title,
                        url=r.url,
                        kind=kind,
                        source="web",
                        snippet=r.snippet[:300],
                        arxiv_id=arxiv_id,
                    )
                )
            return out

        async def all_papers() -> list[list[MethodCandidate]]:
            # arXiv asks clients to pace requests, so its searches run one at a time.
            return [await papers(q) for q in plan.method_queries]

        paper_lists, web_lists = await asyncio.gather(
            all_papers(), asyncio.gather(*(web(q) for q in plan.method_queries))
        )
        # Interleave sources (paper, web, paper, ...) so no single source crowds the list.
        merged: list[MethodCandidate] = []
        for pair in zip(paper_lists, web_lists, strict=True):
            for items in zip(*pair, strict=False):
                merged.extend(items)
            longer = pair[0] if len(pair[0]) > len(pair[1]) else pair[1]
            merged.extend(longer[min(len(pair[0]), len(pair[1])) :])

        seen: set[str] = set()
        candidates = []
        for cand in merged:
            key = cand.arxiv_id or cand.url.rstrip("/").lower()
            if key not in seen:
                seen.add(key)
                candidates.append(cand)
        return {
            "method_candidates": candidates[: settings.policy.max_method_candidates],
            "errors": errors,
        }

    return method_scout
