"""Verify + link: code-only checks for every candidate, and connections from Hub metadata.

1. Models: existence, licence, safety, task, hardware, language (tools/checks/open_weight.py).
2. Link: the chosen models' card tags name datasets they were trained on (`dataset:`) and papers
   that describe them (`arxiv:`). Those are added as candidates, so the kit is connected.
3. Datasets: existence, licence, viewer, modality, language (tools/checks/datasets.py).
4. Methods: arXiv ids confirmed via the arXiv API; every other URL must resolve right now.
No LLM is involved: an item that cannot be verified never reaches the starter kit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.graph.deps import Deps
from app.graph.state import HubScoutState
from app.schemas import (
    DatasetPick,
    MethodCandidate,
    MethodPick,
    OpenWeightCandidate,
    RejectedCandidate,
    ScoutCandidate,
)
from app.tools.checks.datasets import check_dataset
from app.tools.checks.open_weight import check_candidate


async def bounded_gather[T, R](
    items: list[T], fn: Callable[[T], Awaitable[R]], limit: int
) -> list[R]:
    sem = asyncio.Semaphore(limit)

    async def run(item: T) -> R:
        async with sem:
            return await fn(item)

    return await asyncio.gather(*(run(i) for i in items))


def _dedupe(cands: list[ScoutCandidate]) -> list[ScoutCandidate]:
    seen: set[str] = set()
    out = []
    for c in cands:
        if c.repo_id.lower() not in seen:
            seen.add(c.repo_id.lower())
            out.append(c)
    return out


def make_verify_node(deps: Deps) -> Any:
    policy = deps.settings.policy

    async def verify(state: HubScoutState) -> dict[str, Any]:
        constraints, plan = state["constraints"], state["plan"]
        rejected: list[RejectedCandidate] = []
        checked = 0

        # 1. Models ---------------------------------------------------------------------
        model_cands = _dedupe(state.get("model_candidates", []))
        model_facts = await bounded_gather(
            [c.repo_id for c in model_cands], deps.hub.model_facts, policy.scout_concurrency
        )
        checked += len(model_facts)
        models: list[OpenWeightCandidate] = []
        for cand, facts in zip(model_cands, model_facts, strict=True):
            verdict = check_candidate(facts, constraints, plan.hf_task, policy, cand.why)
            if isinstance(verdict, OpenWeightCandidate):
                models.append(verdict)
            else:
                rejected.append(verdict)
        models.sort(key=lambda m: m.fit_score, reverse=True)
        top_models = models[: policy.kit_items_per_section]

        # 2. Links from Hub metadata ----------------------------------------------------
        dataset_cands = list(state.get("dataset_candidates", []))
        known = {c.repo_id.lower() for c in dataset_cands}
        for model in top_models:
            for ds in model.trained_on:
                if ds.lower() not in known:
                    known.add(ds.lower())
                    dataset_cands.append(
                        ScoutCandidate(repo_id=ds, why=f"{model.repo_id} was trained on it")
                    )
        method_cands = list(state.get("method_candidates", []))
        known_papers = {c.arxiv_id for c in method_cands if c.arxiv_id}
        linked_ids = [p for m in top_models for p in m.papers if p not in known_papers]
        linked_ids = list(dict.fromkeys(linked_ids))[: policy.kit_items_per_section]

        # 3. Datasets --------------------------------------------------------------------
        dataset_cands = _dedupe(dataset_cands)
        ds_facts = await bounded_gather(
            [c.repo_id for c in dataset_cands], deps.hub.dataset_facts, policy.scout_concurrency
        )
        checked += len(ds_facts)
        datasets: list[DatasetPick] = []
        for dcand, dfacts in zip(dataset_cands, ds_facts, strict=True):
            dverdict = check_dataset(dfacts, constraints, plan.hf_task, policy, dcand.why)
            if isinstance(dverdict, DatasetPick):
                used_by = [
                    m.repo_id
                    for m in top_models
                    if dverdict.repo_id.lower() in {d.lower() for d in m.trained_on}
                ]
                datasets.append(dverdict.model_copy(update={"used_by": used_by}))
            else:
                rejected.append(dverdict)
        # Connected first (a chosen model was trained on it), then by popularity.
        datasets.sort(key=lambda d: (bool(d.used_by), d.score), reverse=True)

        # 4. Methods ---------------------------------------------------------------------
        now = datetime.now(UTC)
        errors: list[str] = []
        try:
            linked_papers = await deps.arxiv.get(linked_ids) if linked_ids else []
        except Exception as exc:
            linked_papers = []
            errors.append(f"arXiv lookup for linked papers failed ({type(exc).__name__})")
        linked_cands = [
            MethodCandidate(
                title=paper.title,
                url=paper.url,
                kind="paper",
                source="hub-metadata",
                snippet=paper.summary[:300],
                arxiv_id=paper.arxiv_id,
                published=paper.published,
            )
            for paper in linked_papers
        ]
        method_cands = [*linked_cands, *method_cands]
        # arXiv ids that came from the web (not from arXiv itself) must be confirmed.
        web_arxiv = [m.arxiv_id for m in method_cands if m.arxiv_id and m.source == "web"]
        try:
            confirmed = (
                {p.arxiv_id for p in await deps.arxiv.get(web_arxiv)} if web_arxiv else set()
            )
        except Exception:
            # arXiv API unavailable: confirm each paper by its abstract page resolving instead.
            pages = await bounded_gather(
                [f"https://arxiv.org/abs/{i}" for i in web_arxiv],
                deps.links.check,
                policy.scout_concurrency,
            )
            confirmed = {i for i, s in zip(web_arxiv, pages, strict=True) if s.ok}
        needs_http = [c for c in method_cands if not c.arxiv_id]
        statuses = await bounded_gather(
            [c.url for c in needs_http], deps.links.check, policy.scout_concurrency
        )
        http_ok = {s.url: s for s in statuses}
        checked += len(method_cands)

        methods: list[MethodPick] = []
        for mc in method_cands:
            if mc.arxiv_id:
                ok = mc.source in ("arxiv", "hub-metadata") or mc.arxiv_id in confirmed
                reason = "arXiv id could not be confirmed"
            else:
                status = http_ok[mc.url]
                ok, reason = status.ok, status.reason
            if not ok:
                rejected.append(
                    RejectedCandidate(
                        repo_id=mc.url, stage="existence", kind="method", reasons=[reason]
                    )
                )
                continue
            describes = [m.repo_id for m in top_models if mc.arxiv_id in m.papers]
            methods.append(
                MethodPick(
                    title=mc.title,
                    url=mc.url,
                    kind=mc.kind,
                    source=mc.source,
                    why=mc.snippet[:200] or mc.title,
                    arxiv_id=mc.arxiv_id,
                    published=mc.published,
                    describes=describes,
                    retrieved_at=now,
                )
            )

        return {
            "verified_models": models,
            "verified_datasets": datasets,
            "verified_methods": methods,
            "rejected": rejected,
            "sources_checked": checked,
            "errors": errors,
        }

    return verify
