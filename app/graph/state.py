"""Graph state for the main HubScout graph."""

from __future__ import annotations

import operator
from collections.abc import Callable
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


def unique_merge(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer: append new items, keep first-seen order, no duplicates."""
    merged = list(left or [])
    for item in right or []:
        if item not in merged:
            merged.append(item)
    return merged


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
    models_used: Annotated[list[str], unique_merge]
    errors: Annotated[list[str], operator.add]


def ask_until_answered(payload: dict[str, Any], interrupt_fn: Callable[[Any], Any]) -> Any:
    """Interrupt until the human sends a non-empty answer.

    Studio's resume box starts as "" — resuming without typing inside the quotes must re-ask,
    not silently count as an answer. Repeated interrupt() calls in one node are matched to
    resume values in order, so earlier (empty) answers replay correctly after each resume.
    """
    answer = interrupt_fn(payload)
    while not as_text(answer):
        answer = interrupt_fn(
            {**payload, "error": "The answer was empty. Type it inside the quotes, then Resume."}
        )
    return answer


def as_text(value: Any) -> str:
    """Normalise a human resume value (string, dict or anything) into text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "feedback", "response"):
            if isinstance(value.get(key), str):
                return str(value[key]).strip()
    return str(value)
