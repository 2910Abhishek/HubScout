"""Aggregator: build the validated Blueprint. Facts come from code; the LLM writes the prose."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.graph import prompts
from app.graph.deps import Deps
from app.graph.state import HubScoutState
from app.graph.usage import ModelUsageRecorder
from app.schemas import AggregatorNarrative, Blueprint, OpenWeightCandidate

API_PENDING = "The hosted-API path (API model scout, pricing) is not implemented in this POC yet."


def pipeline_mermaid(pick: OpenWeightCandidate, task: str) -> str:
    return (
        "flowchart LR\n"
        f'  A["Input data"] --> B["{pick.repo_id}<br/>{pick.precision}, ~{pick.est_vram_gb} GB"]\n'
        f'  B --> C["{task} output"]'
    )


def make_aggregator_node(deps: Deps) -> Any:
    async def aggregate(state: HubScoutState, config: RunnableConfig) -> dict[str, Any]:
        constraints = state["constraints"]
        approved = state.get("approved", [])
        rejected = state.get("rejected", [])
        notes = list(state.get("assumptions", []))
        models_used: list[str] = []
        mode = constraints.deployment_mode

        if mode in ("api", "compare"):
            notes.append(API_PENDING)

        pick = approved[0] if approved and mode != "api" else None
        alternatives = approved[1:] if pick else []
        if pick is None:
            if mode != "api":
                notes.append(
                    f"No open-weight candidate passed the checks ({len(rejected)} rejected); "
                    "see `rejected` for the reasons."
                )
            summary = "No recommendation could be verified for these constraints."
            risks: list[str] = []
            reasons: list[str] = []
        else:
            facts = {
                "constraints": constraints.model_dump(mode="json"),
                "verified_candidates": [c.model_dump(mode="json") for c in approved],
            }
            recorder = ModelUsageRecorder()
            llm = deps.llm("strong", schema=AggregatorNarrative).with_config(callbacks=[recorder])
            narrative: AggregatorNarrative = await llm.ainvoke(
                [SystemMessage(prompts.AGGREGATOR), HumanMessage(json.dumps(facts, indent=2))],
                config,
            )
            models_used = recorder.models
            summary, risks, reasons = (
                narrative.recommendation_summary,
                narrative.risks,
                narrative.pick_reasons,
            )
            pick = pick.model_copy(update={"reasons": [*reasons, *pick.reasons]})

        chosen = [c for c in ([pick] if pick else []) + alternatives]
        blueprint = Blueprint(
            constraints=constraints,
            open_weight_pick=pick,
            open_weight_alternatives=alternatives,
            rejected=rejected,
            recommendation_summary=summary,
            risks=risks,
            licensing_notes=[f"{c.repo_id}: {c.licence}" for c in chosen],
            pipeline_mermaid=pipeline_mermaid(pick, state["plan"].hf_task) if pick else None,
            sources=[s for c in chosen for s in c.sources],
            notes=notes,
            models_used=sorted({*state.get("models_used", []), *models_used}),
            created_at=datetime.now(UTC),
        )
        return {"blueprint": blueprint, "models_used": models_used}

    return aggregate
