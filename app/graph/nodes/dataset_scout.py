"""Dataset scout: code-driven Hub dataset search through the official HF MCP server.

No LLM needed: the planner already wrote the dataset queries, and every candidate is verified
by code afterwards, so searching broadly is cheap and safe. Datasets that chosen models were
trained on are added later from Hub metadata (see the verify node).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

from app.graph.deps import Deps
from app.graph.nodes.scout import MAX_SEARCH_RESULTS, repo_ids_in_text, tool_text
from app.graph.state import HubScoutState
from app.schemas import ScoutCandidate
from app.tools.checks.languages import normalize

logger = logging.getLogger(__name__)
NO_RESULTS = "No repositories found"


async def _search(tool: BaseTool, query: str | None, filters: list[str]) -> str:
    args: dict[str, Any] = {
        "repo_types": ["dataset"],
        "filters": filters,
        "sort": "downloads",
        "limit": MAX_SEARCH_RESULTS,
    }
    if query:
        args["query"] = query
    try:
        return tool_text(await tool.ainvoke(args))
    except Exception as exc:  # a failed search is just an empty result
        logger.warning("dataset search failed (%s): %s", args, exc)
        return ""


def make_dataset_scout_node(deps: Deps) -> Any:
    policy = deps.settings.policy

    async def dataset_scout(state: HubScoutState) -> dict[str, Any]:
        constraints, plan = state["constraints"], state["plan"]
        try:
            tools = {tool.name: tool for tool in await deps.load_tools()}
        except Exception as exc:  # MCP server unreachable: an empty section, not a crash
            logger.warning("HF MCP tools unavailable: %s", exc)
            tools = {}
        search = tools.get("hub_repo_search")
        if search is None:
            return {"dataset_candidates": [], "errors": ["dataset scout: Hub search unavailable"]}

        task_filter = f"task_categories:{plan.hf_task}"
        lang_filters = [f"language:{normalize(lang)}" for lang in constraints.languages[:2]]
        attempts: list[tuple[str | None, list[str]]] = []
        for q in plan.dataset_queries:
            attempts += [(q, [task_filter]), (q, [])]
        # Grounded fallbacks: the most-downloaded datasets for the task (and languages).
        attempts += [(None, [task_filter, *lang_filters]), (None, [task_filter])]

        candidates: list[ScoutCandidate] = []
        seen: set[str] = set()
        for query, filters in attempts:
            if len(candidates) >= policy.max_dataset_candidates:
                break
            text = await _search(search, query, filters)
            if not text or NO_RESULTS in text:
                continue
            label = f"'{query}'" if query else "top datasets"
            where = f" ({', '.join(filters)})" if filters else ""
            for repo in repo_ids_in_text(text):
                if repo.lower() not in seen:
                    seen.add(repo.lower())
                    candidates.append(
                        ScoutCandidate(repo_id=repo, why=f"Hub dataset search {label}{where}")
                    )
        return {"dataset_candidates": candidates[: policy.max_dataset_candidates]}

    return dataset_scout
