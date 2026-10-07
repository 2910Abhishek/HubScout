"""Planner (LLM) and plan review (human-approval interrupt, no LLM)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.graph import prompts
from app.graph.deps import Deps
from app.graph.state import HubScoutState, as_text, ask_until_answered
from app.graph.usage import ModelUsageRecorder
from app.schemas import PlanReview, ResearchPlan

_YES = {"y", "yes", "ok", "okay", "approve", "approved", "true", "go", "lgtm"}


def parse_review(value: Any) -> PlanReview:
    if isinstance(value, bool):
        return PlanReview(approved=value)
    if isinstance(value, dict) and "approved" in value:
        return PlanReview.model_validate(value)
    text = as_text(value)
    if text.lower().rstrip(".!") in _YES:
        return PlanReview(approved=True)
    return PlanReview(approved=False, feedback=text or None)


def make_planner_node(deps: Deps) -> Any:
    async def plan(state: HubScoutState, config: RunnableConfig) -> dict[str, Any]:
        constraints = state["constraints"]
        feedback = state.get("plan_feedback")
        brief = f"Constraints:\n{constraints.model_dump_json(indent=2)}"
        if feedback:
            brief += f"\n\nReviewer feedback on the previous plan:\n{feedback}"
        recorder = ModelUsageRecorder()
        llm = deps.llm("strong", schema=ResearchPlan).with_config(callbacks=[recorder])
        research_plan: ResearchPlan = await llm.ainvoke(
            [SystemMessage(prompts.PLANNER), HumanMessage(brief)], config
        )
        return {"plan": research_plan, "plan_approved": False, "models_used": recorder.models}

    return plan


def make_review_node(deps: Deps) -> Any:
    def review_plan(state: HubScoutState) -> dict[str, Any]:
        revisions = state.get("plan_revisions", 0)
        answer = ask_until_answered(
            {
                "type": "plan_approval",
                "plan": state["plan"].model_dump(),
                "how_to_answer": 'Type "yes" inside the quotes of the Resume box to approve, or '
                "type feedback to get a revised plan, then click Resume.",
            },
            interrupt,
        )
        review = parse_review(answer)
        if review.approved or revisions >= deps.settings.policy.max_plan_revisions:
            return {"plan_approved": True, "plan_feedback": None}
        return {
            "plan_approved": False,
            "plan_feedback": review.feedback,
            "plan_revisions": revisions + 1,
        }

    return review_plan
