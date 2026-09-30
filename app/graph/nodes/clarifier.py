"""Clarifier: extract constraints and ask only what is missing (licence use, task, hardware).

Two nodes so that resuming an interrupt never re-runs an LLM call:
- `clarify` (LLM) extracts a draft and decides, in code, which questions are still needed.
- `ask_user` only calls `interrupt()` and records the answer; the graph loops back to `clarify`.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.graph import prompts
from app.graph.deps import Deps
from app.graph.state import HubScoutState, as_text, ask_until_answered
from app.graph.usage import ModelUsageRecorder
from app.schemas import ConstraintDraft, Constraints

Q_COMMERCIAL = "Is this for commercial use (yes/no)?"
Q_TASK = "Is the task about text, speech, or vision?"
Q_HARDWARE = "What hardware will run the model? For example '16 GB GPU' or 'CPU only'."


_FAMILY_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("speech", re.compile(r"speech|audio|\basr\b|transcri|voice|spoken|call recording|tts")),
    ("vision", re.compile(r"image|photo|video|vision|picture|object detection|\bocr\b|camera")),
    ("text", re.compile(r"\btext\b|ticket|document|sentiment|classif|summar|translat|chat|email")),
]


def infer_task_family(text: str) -> str | None:
    """Deterministic backup when the LLM leaves task_family empty; first match wins."""
    lowered = text.lower()
    for family, pattern in _FAMILY_KEYWORDS:
        if pattern.search(lowered):
            return family
    return None


def missing_questions(draft: ConstraintDraft) -> list[str]:
    """Deterministic: which required facts are still unknown (at most one round of 1-3)."""
    questions: list[str] = []
    if draft.commercial_use is None:
        questions.append(Q_COMMERCIAL)
    if draft.task_family is None:
        questions.append(Q_TASK)
    if draft.gpu_vram_gb is None and not draft.cpu_only:
        questions.append(Q_HARDWARE)
    return questions


def finalize(draft: ConstraintDraft, request: str) -> tuple[Constraints, list[str]]:
    """Fill anything still missing with conservative defaults and say so."""
    assumptions: list[str] = []
    commercial = draft.commercial_use
    if commercial is None:
        commercial = True
        assumptions.append("commercial use not stated; assumed yes (stricter licence rules)")
    family = draft.task_family
    if family is None:
        family = "text"
        assumptions.append("task family unclear; assumed text")
    cpu_only = bool(draft.cpu_only)
    if draft.gpu_vram_gb is None and not cpu_only:
        cpu_only = True
        assumptions.append("hardware not given; assumed CPU only")
    constraints = Constraints(
        task_family=family,
        task_description=draft.task_description or request,
        commercial_use=commercial,
        languages=draft.languages,
        gpu_vram_gb=None if cpu_only else draft.gpu_vram_gb,
        cpu_only=cpu_only,
        needs_finetuning=bool(draft.needs_finetuning),
        licence_policy=draft.licence_policy,
        monthly_volume=draft.monthly_volume,
        monthly_budget_usd=draft.monthly_budget_usd,
        data_can_leave_org=draft.data_can_leave_org,
        max_latency_ms=draft.max_latency_ms,
    )
    return constraints, assumptions


def make_clarify_node(deps: Deps) -> Any:
    async def clarify(state: HubScoutState, config: RunnableConfig) -> dict[str, Any]:
        log = state.get("clarification_log", [])
        convo = state["request"] + ("\n\n" + "\n".join(log) if log else "")
        recorder = ModelUsageRecorder()
        llm = deps.llm("strong", schema=ConstraintDraft).with_config(callbacks=[recorder])
        draft: ConstraintDraft = await llm.ainvoke(
            [SystemMessage(prompts.CLARIFIER), HumanMessage(convo)], config
        )
        # Only the user's own words: the log also contains our questions ("text, speech...").
        user_text = " ".join(
            [state["request"], *(e.split("\nA: ", 1)[-1] for e in log if "\nA: " in e)]
        )
        if draft.task_family is None and (family := infer_task_family(user_text)):
            draft = draft.model_copy(update={"task_family": family})
        questions = missing_questions(draft)
        rounds = state.get("clarify_rounds", 0)
        update: dict[str, Any] = {"draft": draft, "models_used": recorder.models}
        if questions and rounds < deps.settings.policy.max_clarify_rounds:
            return {**update, "pending_questions": questions}
        constraints, assumptions = finalize(draft, state["request"])
        return {
            **update,
            "pending_questions": [],
            "constraints": constraints,
            "assumptions": assumptions,
        }

    return clarify


def ask_user(state: HubScoutState) -> dict[str, Any]:
    questions = state.get("pending_questions", [])
    answer = ask_until_answered(
        {
            "type": "clarification",
            "questions": questions,
            "how_to_answer": "Answer all questions in one line inside the quotes of the Resume "
            'box, e.g. "commercial, speech, 16 GB GPU", then click Resume.',
        },
        interrupt,
    )
    entry = "Q: " + " / ".join(questions) + "\nA: " + as_text(answer)
    return {
        "clarification_log": [entry],
        "clarify_rounds": state.get("clarify_rounds", 0) + 1,
        "pending_questions": [],
    }
