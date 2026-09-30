"""Write: build the validated StarterKit and render the README. Facts come from code.

LLM calls (both optional, with code fallbacks): choose method resources among VERIFIED items,
and write the TL;DR + fit story from verified facts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import PROJECT_ROOT
from app.graph import prompts
from app.graph.deps import Deps
from app.graph.state import HubScoutState
from app.graph.usage import ModelUsageRecorder
from app.report.readme import quick_start, render_readme
from app.schemas import (
    DatasetPick,
    KitNarrative,
    MethodPick,
    MethodSelection,
    OpenWeightCandidate,
    RejectedCandidate,
    StarterKit,
)

logger = logging.getLogger(__name__)


def rank_models(
    models: list[OpenWeightCandidate], dataset_ids: set[str]
) -> list[OpenWeightCandidate]:
    """Fit score, with a small boost for models connected to a verified dataset."""

    def key(m: OpenWeightCandidate) -> float:
        linked = any(d.lower() in dataset_ids for d in m.trained_on)
        return m.fit_score + (0.1 if linked else 0.0)

    return sorted(models, key=key, reverse=True)


def default_method_order(methods: list[MethodPick]) -> list[MethodPick]:
    """Fallback when the LLM can't choose: linked papers, other papers, repos, guides."""
    rank = {"hub-metadata": 0, "arxiv": 1, "web": 2}
    return sorted(methods, key=lambda m: (rank[m.source], m.kind == "guide"))


def gaps_for(
    models: list[Any], datasets: list[Any], methods: list[Any], rejected: list[RejectedCandidate]
) -> list[str]:
    """Honest gaps: say what is missing and the nearest rejected option, never pad."""
    gaps: list[str] = []
    for label, items, kind in (
        ("model", models, "model"),
        ("dataset", datasets, "dataset"),
        ("method resource", methods, "method"),
    ):
        if items:
            continue
        near = next((r for r in rejected if r.kind == kind and r.stage == "constraints"), None)
        hint = f" Nearest option: {near.repo_id} ({near.reasons[0]})." if near else ""
        gaps.append(f"No {label} passed verification for these constraints.{hint}")
    if models and datasets and not any(d.used_by for d in datasets):
        gaps.append(
            "The Hub lists no training datasets for the chosen models among the verified "
            "datasets; the datasets are recommended on their own merits."
        )
    return gaps


def save_readme(text: str, task: str, output_dir: str) -> str | None:
    """Save the README under output_dir; an empty output_dir disables saving."""
    if not output_dir:
        return None
    folder = Path(output_dir) if Path(output_dir).is_absolute() else PROJECT_ROOT / output_dir
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:60] or "starter-kit"
    path = folder / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{slug}.md"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("could not save README: %s", exc)
        return None
    return str(path)


def make_write_node(deps: Deps) -> Any:
    policy = deps.settings.policy
    n = policy.kit_items_per_section

    async def write(state: HubScoutState, config: RunnableConfig) -> dict[str, Any]:
        constraints, plan = state["constraints"], state["plan"]
        rejected = state.get("rejected", [])
        datasets: list[DatasetPick] = state.get("verified_datasets", [])[:n]
        models = rank_models(
            state.get("verified_models", []), {d.repo_id.lower() for d in datasets}
        )[:n]
        verified_methods: list[MethodPick] = state.get("verified_methods", [])
        recorder = ModelUsageRecorder()
        errors: list[str] = []

        # Methods: the LLM chooses among verified items only (by number); code validates.
        linked = [m for m in verified_methods if m.describes]
        others = [m for m in verified_methods if not m.describes]
        methods = linked[:n]
        if others and len(methods) < n:
            numbered = "\n".join(
                f"{i}. [{m.kind}] {m.title} :: {m.why[:200]}" for i, m in enumerate(others)
            )
            ask = (
                f"Task: {constraints.task_description}\nConstraints: "
                f"{constraints.model_dump_json()}\n\nResults:\n{numbered}"
            )
            system = prompts.METHOD_SELECT.format(max_items=n - len(methods))
            try:
                llm = deps.llm("cheap", schema=MethodSelection).with_config(callbacks=[recorder])
                sel: MethodSelection = await llm.ainvoke(
                    [SystemMessage(system), HumanMessage(ask)], config
                )
                picked: list[MethodPick] = []
                for idx, reason in zip(
                    sel.picks, [*sel.reasons, *[""] * len(sel.picks)], strict=False
                ):
                    if 0 <= idx < len(others) and others[idx] not in picked:
                        pick = others[idx]
                        picked.append(pick.model_copy(update={"why": reason or pick.why}))
            except Exception as exc:
                logger.warning("method selection failed (%s); using default order", exc)
                errors.append(f"method selection failed ({type(exc).__name__}); default order")
                picked = []
            if not picked:  # nothing valid chosen: verified items in a sensible default order
                picked = default_method_order(others)
            methods += picked[: n - len(methods)]

        gaps = gaps_for(models, datasets, methods, rejected)
        kit_ids = {d.repo_id.lower() for d in datasets}
        facts = {
            "constraints": constraints.model_dump(mode="json"),
            # Only relationships to items that are in the kit: unverified names must not leak.
            "models": [
                {
                    **m.model_dump(mode="json", include={"repo_id", "licence", "params_billion",
                    "est_vram_gb", "precision", "reasons"}),
                    "trained_on_kit_datasets": [d for d in m.trained_on if d.lower() in kit_ids],
                }
                for m in models
            ],
            "datasets": [
                d.model_dump(mode="json", include={"repo_id", "licence", "num_rows", "splits",
                "used_by", "why"})
                for d in datasets
            ],
            "methods": [
                m.model_dump(mode="json", include={"title", "kind", "describes", "why"})
                for m in methods
            ],
            # Only whether a section is empty; never the rejected "nearest option" hints.
            "empty_sections": [s for s, items in (("models", models), ("datasets", datasets),
                               ("methods", methods)) if not items],
        }  # fmt: skip
        tldr, fit_story = _fallback_narrative(models, datasets, methods)
        if models or datasets or methods:
            try:
                llm = deps.llm("strong", schema=KitNarrative).with_config(callbacks=[recorder])
                narrative: KitNarrative = await llm.ainvoke(
                    [SystemMessage(prompts.KIT_NARRATIVE), HumanMessage(json.dumps(facts))],
                    config,
                )
                tldr, fit_story = narrative.tldr, narrative.fit_story
            except Exception as exc:
                logger.warning("narrative failed (%s); using code summary", exc)
                errors.append(f"narrative failed ({type(exc).__name__}); code summary used")

        kit = StarterKit(
            constraints=constraints,
            tldr=tldr,
            fit_story=fit_story,
            models=models,
            datasets=datasets,
            methods=methods,
            rejected=rejected,
            gaps=gaps,
            assumptions=state.get("assumptions", []),
            models_used=list(dict.fromkeys([*state.get("models_used", []), *recorder.models])),
            sources_checked=state.get("sources_checked", 0),
            created_at=datetime.now(UTC),
        )
        kit = kit.model_copy(update={"quick_start": quick_start(kit, plan.hf_task)})
        readme = render_readme(kit)
        # File I/O off the event loop (the LangGraph server rejects blocking calls in async nodes).
        path = await asyncio.to_thread(
            save_readme, readme, constraints.task_description, policy.output_dir
        )
        return {
            "starter_kit": kit,
            "readme": readme,
            "readme_path": path,
            "models_used": recorder.models,
            "errors": errors,
        }

    return write


def _fallback_narrative(
    models: list[OpenWeightCandidate], datasets: list[DatasetPick], methods: list[MethodPick]
) -> tuple[str, str]:
    parts = []
    if models:
        m = models[0]
        parts.append(f"Start with {m.repo_id} ({m.licence}, ~{m.est_vram_gb} GB at {m.precision}).")
    if datasets:
        parts.append(f"Train or evaluate on {datasets[0].repo_id} ({datasets[0].licence}).")
    if methods:
        parts.append(f"For the approach, read '{methods[0].title}'.")
    tldr = " ".join(parts) or "No item could be verified for these constraints."
    story = (
        f"{models[0].repo_id} was trained on {datasets[0].repo_id}."
        if models and datasets and datasets[0].used_by
        else "These items were chosen independently; the Hub records no direct link between them."
    )
    return tldr, story
