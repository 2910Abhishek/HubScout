"""Open-weight scout: an LLM that searches the Hub through the official HF MCP server.

A small, explicit tool loop (instead of a prebuilt agent) so every step is visible in Studio
under `scout_messages`, the number of LLM calls is bounded, and tool output is size-capped.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from app.graph import prompts
from app.graph.deps import Deps
from app.graph.state import HubScoutState
from app.graph.usage import ModelUsageRecorder
from app.schemas import ScoutCandidate, ScoutReport

logger = logging.getLogger(__name__)
# Keep the tool conversation small: local fallback models have a ~4K-token context on CPU.
MAX_TOOL_CHARS = 4000
MAX_SEARCH_RESULTS = 8
# Repo-like ids, but not dataset tags ("dataset:org/name"), base-model tags or URL fragments.
_REPO_ID = re.compile(r"(?<![\w:/.-])([A-Za-z0-9][\w.-]{1,95}/[A-Za-z0-9][\w.-]{1,95})\b")
_NOT_REPOS = ("huggingface.co", "hf.co", "http", "www.")


def repo_ids_in_tool_results(messages: list[AnyMessage], limit: int) -> list[ScoutCandidate]:
    """Deterministic fallback: repo ids that literally appear in tool output, in order.

    Safe because every candidate is verified against the Hub by the checker afterwards.
    """
    found: list[ScoutCandidate] = []
    seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for match in _REPO_ID.findall(str(msg.content)):
            key = match.strip(".").lower()
            if key in seen or any(bad in key for bad in _NOT_REPOS):
                continue
            seen.add(key)
            found.append(
                ScoutCandidate(repo_id=match.strip("."), why="found in Hub search results")
            )
            if len(found) >= limit:
                return found
    return found


def _truncate(content: Any) -> str:
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + "\n...[truncated]"


async def _run_tool(tools: dict[str, BaseTool], call: dict[str, Any]) -> ToolMessage:
    tool = tools.get(call["name"])
    if tool is None:
        return ToolMessage(
            f"Unknown tool {call['name']!r}", tool_call_id=call["id"], status="error"
        )
    args = dict(call["args"])
    if call["name"] == "hub_repo_search":
        args["limit"] = min(int(args.get("limit") or MAX_SEARCH_RESULTS), MAX_SEARCH_RESULTS)
    try:
        result = await tool.ainvoke(args)
    except Exception as exc:  # tool errors go back to the model, not up the graph
        logger.warning("scout tool %s failed: %s", call["name"], exc)
        return ToolMessage(f"Tool error: {exc}", tool_call_id=call["id"], status="error")
    return ToolMessage(_truncate(result), tool_call_id=call["id"], name=call["name"])


async def grounded_search(
    tools: dict[str, BaseTool], hf_task: str, languages: list[str]
) -> list[AnyMessage]:
    """Deterministic backup search when the LLM's searches found nothing: top models for the
    task (and languages), by downloads. Recorded as a normal tool exchange for Studio."""
    messages: list[AnyMessage] = []
    for filters in ([hf_task, *languages], [hf_task]) if languages else ([hf_task],):
        args = {
            "filters": filters,
            "sort": "downloads",
            "limit": MAX_SEARCH_RESULTS,
            "repo_types": ["model"],
        }
        call = {"name": "hub_repo_search", "args": args, "id": f"fallback-{len(messages)}"}
        messages.append(AIMessage(content="(fallback search by HubScout code)", tool_calls=[call]))
        result = await _run_tool(tools, call)
        messages.append(result)
        if "No repositories found" not in str(result.content) and result.status != "error":
            break
    return messages


def make_scout_node(deps: Deps) -> Any:
    policy = deps.settings.policy

    async def scout(state: HubScoutState, config: RunnableConfig) -> dict[str, Any]:
        constraints, plan = state["constraints"], state["plan"]
        tools = await deps.load_tools()
        by_name = {tool.name: tool for tool in tools}
        hw = "CPU only" if constraints.cpu_only else f"{constraints.gpu_vram_gb} GB GPU"
        brief = (
            f"Task: {constraints.task_description}\n"
            f"Pipeline tag: {plan.hf_task}\n"
            f"Search queries to try: {plan.search_queries}\n"
            f"Hardware limit: {hw}\n"
            f"Languages: {constraints.languages or 'not specified'}\n"
            f"Commercial use: {constraints.commercial_use}\n"
            f"Selection criteria: {plan.selection_criteria}"
        )
        messages: list[AnyMessage] = [SystemMessage(prompts.SCOUT), HumanMessage(brief)]
        recorder = ModelUsageRecorder()
        agent = deps.llm("cheap", tools=tools).with_config(callbacks=[recorder])

        errors: list[str] = []
        for _ in range(policy.scout_max_tool_rounds):
            try:
                reply = await agent.ainvoke(messages, config)
            except Exception as exc:  # degrade: keep whatever the searches found so far
                logger.warning("scout step failed (%s); continuing with results so far", exc)
                errors.append(f"scout step failed ({type(exc).__name__}); used results so far")
                break
            if not isinstance(reply, AIMessage):
                break
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                messages.append(await _run_tool(by_name, dict(call)))

        if not repo_ids_in_tool_results(messages, 1):
            messages.extend(await grounded_search(by_name, plan.hf_task, constraints.languages))

        ask = HumanMessage(prompts.SCOUT_REPORT.format(max_candidates=policy.max_candidates))
        reporter = deps.llm("cheap", schema=ScoutReport).with_config(callbacks=[recorder])
        try:
            report: ScoutReport = await reporter.ainvoke([*messages, ask], config)
        except Exception as exc:  # graceful degradation: fall back to parsing tool output
            logger.warning("scout report failed (%s); using repo ids from tool results", exc)
            errors.append(f"scout report failed ({type(exc).__name__}); used tool results")
            report = ScoutReport(
                candidates=repo_ids_in_tool_results(messages, policy.max_candidates)
            )

        # The LLM's picks first; remaining slots filled from search results (sorted by
        # downloads). Everything is verified by the checker, so breadth is cheap and safe.
        pool = [*report.candidates, *repo_ids_in_tool_results(messages, policy.max_candidates)]
        seen: set[str] = set()
        candidates = []
        for cand in pool:
            key = cand.repo_id.strip()
            if key and key.lower() not in seen:
                seen.add(key.lower())
                candidates.append(cand.model_copy(update={"repo_id": key}))
        return {
            "scout_messages": messages[1:],  # skip the long system prompt
            "candidates": candidates[: policy.max_candidates],
            "models_used": recorder.models,
            "errors": errors,
        }

    return scout
