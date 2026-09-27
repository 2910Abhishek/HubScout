"""Assemble the main HubScout graph (registered as "hubscout" in langgraph.json).

clarify ──(questions?)──> ask_user ──> clarify
   │ settled
   ├── api mode ─────────────────────────────────────────> aggregate ──> END
   └──> plan ──> review_plan ──(feedback)──> plan
                      │ approved
                      └──> scout (LLM + HF MCP) ──> check (code) ──> aggregate
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.deps import Deps, default_deps
from app.graph.nodes.aggregator import make_aggregator_node
from app.graph.nodes.checker import make_checker_node
from app.graph.nodes.clarifier import ask_user, make_clarify_node
from app.graph.nodes.planner import make_planner_node, make_review_node
from app.graph.nodes.scout import make_scout_node
from app.graph.state import HubScoutState, InputState


def after_clarify(state: HubScoutState) -> Literal["ask_user", "plan", "aggregate"]:
    if state.get("pending_questions"):
        return "ask_user"
    if state["constraints"].deployment_mode == "api":
        return "aggregate"
    return "plan"


def after_review(state: HubScoutState) -> Literal["plan", "scout"]:
    return "scout" if state.get("plan_approved") else "plan"


def build_graph(
    deps: Deps | None = None, checkpointer: BaseCheckpointSaver[str] | None = None
) -> CompiledStateGraph[HubScoutState, None, InputState, HubScoutState]:
    deps = deps or default_deps()
    builder = StateGraph(HubScoutState, input_schema=InputState)
    builder.add_node("clarify", make_clarify_node(deps))
    builder.add_node("ask_user", ask_user)
    builder.add_node("plan", make_planner_node(deps))
    builder.add_node("review_plan", make_review_node(deps))
    builder.add_node("scout", make_scout_node(deps))
    builder.add_node("check", make_checker_node(deps))
    builder.add_node("aggregate", make_aggregator_node(deps))

    builder.add_edge(START, "clarify")
    builder.add_conditional_edges("clarify", after_clarify)
    builder.add_edge("ask_user", "clarify")
    builder.add_edge("plan", "review_plan")
    builder.add_conditional_edges("review_plan", after_review)
    builder.add_edge("scout", "check")
    builder.add_edge("check", "aggregate")
    builder.add_edge("aggregate", END)
    # `langgraph dev` / LangGraph Server supply their own persistence; tests pass a checkpointer.
    return builder.compile(checkpointer=checkpointer, name="hubscout")


def make_graph() -> CompiledStateGraph[HubScoutState, None, InputState, HubScoutState]:
    """Factory used by langgraph.json: builds real dependencies at server start."""
    return build_graph()
