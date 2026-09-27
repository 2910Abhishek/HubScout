"""Graph state for the main HubScout graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.schemas import (
    Blueprint,
    ConstraintDraft,
    Constraints,
    OpenWeightCandidate,
    RejectedCandidate,
    ResearchPlan,
    ScoutCandidate,
)


class InputState(TypedDict):
    """What you type in Studio: the ML task in plain language."""

    request: str


class HubScoutState(TypedDict, total=False):
    request: str
    # Clarifier
    draft: ConstraintDraft
    pending_questions: list[str]
    clarification_log: Annotated[list[str], operator.add]
    clarify_rounds: int
    assumptions: Annotated[list[str], operator.add]
    constraints: Constraints
    # Planner + human review
    plan: ResearchPlan
    plan_feedback: str | None
    plan_revisions: int
    plan_approved: bool
    # Scout (the full tool-calling conversation is kept so you can inspect it in Studio)
    scout_messages: Annotated[list[AnyMessage], add_messages]
    candidates: list[ScoutCandidate]
    # Checker
    approved: list[OpenWeightCandidate]
    rejected: list[RejectedCandidate]
    # Output
    blueprint: Blueprint
    models_used: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def as_text(value: Any) -> str:
    """Normalise a human resume value (string, dict or anything) into text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "feedback", "response"):
            if isinstance(value.get(key), str):
                return str(value[key]).strip()
    return str(value)
