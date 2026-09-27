"""Open-weight scout: an LLM that searches the Hub through the official HF MCP server.

A small, explicit tool loop (instead of a prebuilt agent) so every step is visible in Studio
under `scout_messages`, the number of LLM calls is bounded, and tool output is size-capped.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from app.graph import prompts
from app.graph.deps import Deps
from app.graph.state import HubScoutState
from app.graph.usage import ModelUsageRecorder
from app.schemas import ScoutReport

logger = logging.getLogger(__name__)
MAX_TOOL_CHARS = 6000


def _truncate(content: Any) -> str:
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + "\n...[truncated]"


async def _run_tool(tools: dict[str, BaseTool], call: dict[str, Any]) -> ToolMessage:
    tool = tools.get(call["name"])
    if tool is None:
        return ToolMessage(
            f"Unknown tool {call['name']!r}", tool_call_id=call["id"], status="error"
        )
    try:
        result = await tool.ainvoke(call["args"])
    except Exception as exc:  # tool errors go back to the model, not up the graph
        logger.warning("scout tool %s failed: %s", call["name"], exc)
        return ToolMessage(f"Tool error: {exc}", tool_call_id=call["id"], status="error")
    return ToolMessage(_truncate(result), tool_call_id=call["id"], name=call["name"])


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

        for _ in range(policy.scout_max_tool_rounds):
            reply = await agent.ainvoke(messages, config)
            if not isinstance(reply, AIMessage):
                break
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                messages.append(await _run_tool(by_name, dict(call)))

        ask = HumanMessage(prompts.SCOUT_REPORT.format(max_candidates=policy.max_candidates))
        reporter = deps.llm("cheap", schema=ScoutReport).with_config(callbacks=[recorder])
        report: ScoutReport = await reporter.ainvoke([*messages, ask], config)

        seen: set[str] = set()
        candidates = []
        for cand in report.candidates:
            key = cand.repo_id.strip()
            if key and key.lower() not in seen:
                seen.add(key.lower())
                candidates.append(cand.model_copy(update={"repo_id": key}))
        return {
            "scout_messages": messages[1:],  # skip the long system prompt
            "candidates": candidates[: policy.max_candidates],
            "models_used": recorder.models,
        }

    return scout
