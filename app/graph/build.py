"""Assemble the main HubScout graph (registered as "hubscout" in langgraph.json).

clarify ──(questions?)──> ask_user ──> clarify
   │ settled
   └──> plan ──> review_plan ──(feedback)──> plan
                      │ approved
                      ├──> model_scout   (LLM + HF MCP) ──┐
                      ├──> dataset_scout (HF MCP, code)  ──┼──> verify (code) ──> write ──> END
                      └──> method_scout  (arXiv + web)   ──┘
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.deps import Deps, default_deps
from app.graph.nodes.clarifier import ask_user, make_clarify_node
from app.graph.nodes.dataset_scout import make_dataset_scout_node
from app.graph.nodes.method_scout import make_method_scout_node
from app.graph.nodes.planner import make_planner_node, make_review_node
from app.graph.nodes.scout import make_model_scout_node
from app.graph.nodes.verify import make_verify_node
from app.graph.nodes.write import make_write_node
from app.graph.state import HubScoutState, InputState

SCOUTS = ("model_scout", "dataset_scout", "method_scout")


def after_clarify(state: HubScoutState) -> Literal["ask_user", "plan"]:
    return "ask_user" if state.get("pending_questions") else "plan"


def after_review(state: HubScoutState) -> Sequence[str]:
    """Approved: run all scouts in parallel. Otherwise: revise the plan."""
    return list(SCOUTS) if state.get("plan_approved") else ["plan"]


def build_graph(
    deps: Deps | None = None, checkpointer: BaseCheckpointSaver[str] | None = None
) -> CompiledStateGraph[HubScoutState, None, InputState, HubScoutState]:
    deps = deps or default_deps()
    builder = StateGraph(HubScoutState, input_schema=InputState)
    builder.add_node("clarify", make_clarify_node(deps))
    builder.add_node("ask_user", ask_user)
    builder.add_node("plan", make_planner_node(deps))
    builder.add_node("review_plan", make_review_node(deps))
    builder.add_node("model_scout", make_model_scout_node(deps))
    builder.add_node("dataset_scout", make_dataset_scout_node(deps))
    builder.add_node("method_scout", make_method_scout_node(deps))
    builder.add_node("verify", make_verify_node(deps))
    builder.add_node("write", make_write_node(deps))

    builder.add_edge(START, "clarify")
    builder.add_conditional_edges("clarify", after_clarify)
    builder.add_edge("ask_user", "clarify")
    builder.add_edge("plan", "review_plan")
    builder.add_conditional_edges("review_plan", after_review, ["plan", *SCOUTS])
    # verify waits for all three scouts (a join), then write finishes the kit.
    builder.add_edge(list(SCOUTS), "verify")
    builder.add_edge("verify", "write")
    builder.add_edge("write", END)
    # `langgraph dev` / LangGraph Server supply their own persistence; tests pass a checkpointer.
    return builder.compile(checkpointer=checkpointer, name="hubscout")


def make_graph() -> CompiledStateGraph[HubScoutState, None, InputState, HubScoutState]:
    """Factory used by langgraph.json: builds real dependencies at server start."""
    return build_graph()
