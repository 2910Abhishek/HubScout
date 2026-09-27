"""Constraint & existence checker: pure code against live Hub data. No LLM."""

from __future__ import annotations

import asyncio
from typing import Any

from app.graph.deps import Deps
from app.graph.state import HubScoutState
from app.schemas import OpenWeightCandidate, RejectedCandidate
from app.tools.checks.open_weight import check_candidate
from app.tools.registries.hub import ModelFacts


def make_checker_node(deps: Deps) -> Any:
    policy = deps.settings.policy

    async def check(state: HubScoutState) -> dict[str, Any]:
        constraints, plan = state["constraints"], state["plan"]
        candidates = state.get("candidates", [])
        limit = asyncio.Semaphore(policy.scout_concurrency)

        async def facts_for(repo_id: str) -> ModelFacts:
            async with limit:
                return await deps.hub.model_facts(repo_id)

        facts = await asyncio.gather(*(facts_for(c.repo_id) for c in candidates))
        approved: list[OpenWeightCandidate] = []
        rejected: list[RejectedCandidate] = []
        for cand, fact in zip(candidates, facts, strict=True):
            verdict = check_candidate(fact, constraints, plan.hf_task, policy, cand.why)
            (approved if isinstance(verdict, OpenWeightCandidate) else rejected).append(verdict)  # type: ignore[arg-type]
        approved.sort(key=lambda c: c.fit_score, reverse=True)
        return {"approved": approved, "rejected": rejected}

    return check
