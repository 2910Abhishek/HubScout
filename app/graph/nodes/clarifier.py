"""Clarifier: extract constraints, ask only what is missing, always settle deployment mode.

Two nodes so that resuming an interrupt never re-runs an LLM call:
- `clarify` (LLM) extracts a draft and decides, in code, which questions are still needed.
- `ask_user` only calls `interrupt()` and records the answer; the graph loops back to `clarify`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.graph import prompts
from app.graph.deps import Deps
from app.graph.state import HubScoutState, as_text
from app.graph.usage import ModelUsageRecorder
from app.schemas import ConstraintDraft, Constraints

Q_MODE = "How do you want to run the model: hosted API, self-hosted open-weight, or compare both?"
Q_COMMERCIAL = "Is this for commercial use (yes/no)?"
Q_TASK = "Is the task about text, speech, or vision?"
Q_HARDWARE = "What hardware will run the model? For example '16 GB GPU' or 'CPU only'."


def missing_questions(draft: ConstraintDraft) -> list[str]:
    """Deterministic: which required facts are still unknown (at most one round of 1-4)."""
    questions: list[str] = []
    if draft.deployment_mode is None:
        questions.append(Q_MODE)
    if draft.commercial_use is None:
        questions.append(Q_COMMERCIAL)
    if draft.task_family is None:
        questions.append(Q_TASK)
    needs_hw = draft.deployment_mode in (None, "open_weight", "compare")
    if needs_hw and draft.gpu_vram_gb is None and not draft.cpu_only:
        questions.append(Q_HARDWARE)
    return questions


def finalize(
    draft: ConstraintDraft, request: str, default_mode: str
) -> tuple[Constraints, list[str]]:
    """Fill anything still missing with conservative defaults and say so."""
    assumptions: list[str] = []
    mode = draft.deployment_mode
    if mode is None:
        mode = "open_weight" if default_mode == "ask" else default_mode  # type: ignore[assignment]
        assumptions.append(f"deployment mode not given; assumed '{mode}'")
    commercial = draft.commercial_use
    if commercial is None:
        commercial = True
        assumptions.append("commercial use not stated; assumed yes (stricter licence rules)")
    family = draft.task_family
    if family is None:
        family = "text"
        assumptions.append("task family unclear; assumed text")
    cpu_only = bool(draft.cpu_only)
    if mode != "api" and draft.gpu_vram_gb is None and not cpu_only:
        cpu_only = True
        assumptions.append("hardware not given; assumed CPU only")
    constraints = Constraints(
        task_family=family,
        task_description=draft.task_description or request,
        deployment_mode=mode,  # type: ignore[arg-type]
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
        questions = missing_questions(draft)
        rounds = state.get("clarify_rounds", 0)
        update: dict[str, Any] = {"draft": draft, "models_used": recorder.models}
        if questions and rounds < deps.settings.policy.max_clarify_rounds:
            return {**update, "pending_questions": questions}
        constraints, assumptions = finalize(
            draft, state["request"], deps.settings.policy.default_deployment_mode
        )
        return {
            **update,
            "pending_questions": [],
            "constraints": constraints,
            "assumptions": assumptions,
        }

    return clarify


def ask_user(state: HubScoutState) -> dict[str, Any]:
    questions = state.get("pending_questions", [])
    answer = interrupt(
        {
            "type": "clarification",
            "questions": questions,
            "how_to_answer": "Resume with one text answer covering all questions.",
        }
    )
    entry = "Q: " + " / ".join(questions) + "\nA: " + as_text(answer)
    return {
        "clarification_log": [entry],
        "clarify_rounds": state.get("clarify_rounds", 0) + 1,
        "pending_questions": [],
    }
