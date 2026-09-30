"""Programmatic checks for open-weight candidates. No LLM involved: pure, testable functions.

Every rule returns human-readable reasons, so rejected candidates can be inspected in Studio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import PolicySettings
from app.schemas import Constraints, OpenWeightCandidate, RejectedCandidate, Source
from app.schemas.candidates import Precision
from app.tools.checks.languages import missing_languages
from app.tools.registries.hub import ModelFacts

BYTES_PER_PARAM: dict[Precision, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}
# Try the most faithful precision first; quantise only if needed to fit.
PRECISION_ORDER: tuple[Precision, ...] = ("bf16", "int8", "int4")
CPU_PRECISION_ORDER: tuple[Precision, ...] = ("fp32", "int8", "int4")


def estimate_vram_gb(params: int, precision: Precision, overhead: float) -> float:
    return round(params * BYTES_PER_PARAM[precision] * overhead / 1e9, 2)


@dataclass(frozen=True)
class FitResult:
    fits: bool
    precision: Precision
    est_gb: float
    limit_gb: float | None
    note: str


def hardware_fit(params: int, constraints: Constraints, policy: PolicySettings) -> FitResult:
    """Pick the best precision that fits the user's GPU (or RAM budget when CPU-only)."""
    if constraints.cpu_only or constraints.gpu_vram_gb is None:
        limit, order, where = policy.cpu_max_model_gb, CPU_PRECISION_ORDER, "system RAM (CPU)"
    else:
        limit, order, where = constraints.gpu_vram_gb, PRECISION_ORDER, "GPU VRAM"
    for precision in order:
        est = estimate_vram_gb(params, precision, policy.vram_overhead_factor)
        if est <= limit:
            return FitResult(
                True, precision, est, limit, f"~{est} GB at {precision} fits {limit} GB {where}"
            )
    smallest = estimate_vram_gb(params, order[-1], policy.vram_overhead_factor)
    return FitResult(
        False,
        order[-1],
        smallest,
        limit,
        f"needs ~{smallest} GB even at {order[-1]}; limit {limit} GB {where}",
    )


def allowed_licences(constraints: Constraints, policy: PolicySettings) -> set[str] | None:
    """None means any known licence is acceptable (non-commercial use, no stated policy)."""
    if constraints.licence_policy:
        return set(constraints.licence_policy)
    if constraints.commercial_use:
        return set(policy.licence_allowlist)
    return None


def popularity(downloads: int) -> float:
    """0..1 on a log scale; 10M downloads/month ~ 1.0."""
    return min(1.0, math.log10(downloads + 1) / 7)


def check_candidate(
    facts: ModelFacts,
    constraints: Constraints,
    hf_task: str | None,
    policy: PolicySettings,
    scout_reason: str,
) -> OpenWeightCandidate | RejectedCandidate:
    if not facts.exists:
        reason = (
            f"could not verify: {facts.error}" if facts.error else "repo does not exist on the Hub"
        )
        return RejectedCandidate(repo_id=facts.repo_id, stage="existence", reasons=[reason])

    reasons: list[str] = []
    notes: list[str] = []

    if facts.gated:
        reasons.append("gated repo: requires accepting terms / manual approval")
    if facts.adapter_only:
        reasons.append("adapter-only repo (e.g. LoRA): needs a separate base model")
    if facts.requires_remote_code:
        reasons.append("requires trust_remote_code (custom code), which HubScout never enables")

    allowed = allowed_licences(constraints, policy)
    if facts.licence is None:
        reasons.append("licence missing from the model card")
    elif allowed is not None and facts.licence not in allowed:
        reasons.append(f"licence '{facts.licence}' not in allowed list {sorted(allowed)}")

    if hf_task and facts.pipeline_tag and facts.pipeline_tag != hf_task:
        reasons.append(f"task mismatch: model is '{facts.pipeline_tag}', need '{hf_task}'")

    fit: FitResult | None = None
    if facts.params is None:
        reasons.append("parameter count unknown (no safetensors metadata), cannot size hardware")
    else:
        fit = hardware_fit(facts.params, constraints, policy)
        if not fit.fits:
            reasons.append(f"too large: {fit.note}")

    missing_langs = missing_languages(constraints.languages, facts.languages)
    if missing_langs and facts.languages and len(missing_langs) == len(constraints.languages):
        reasons.append(f"model card lists none of the required languages {missing_langs}")
    elif missing_langs and facts.languages:
        notes.append(f"model card does not list {missing_langs}")
    elif missing_langs:
        notes.append(f"language support for {missing_langs} not stated on the model card")

    if reasons or fit is None or facts.licence is None:
        return RejectedCandidate(repo_id=facts.repo_id, stage="constraints", reasons=reasons)

    if constraints.cpu_only:
        notes.append("CPU-only inference: expect high latency")
    if facts.params_estimated:
        notes.append("parameter count estimated from weight-file sizes (assumes 16-bit weights)")
    headroom = 1 - fit.est_gb / fit.limit_gb if fit.limit_gb else 0.5
    score = round(0.6 * popularity(facts.downloads) + 0.4 * max(0.0, headroom), 3)
    return OpenWeightCandidate(
        repo_id=facts.repo_id,
        licence=facts.licence,
        params_billion=round(facts.params / 1e9, 3) if facts.params else 0.0,
        est_vram_gb=fit.est_gb,
        precision=fit.precision,
        pipeline_tag=facts.pipeline_tag,
        downloads=facts.downloads,
        fit_score=score,
        reasons=[scout_reason, fit.note],
        notes=notes,
        sources=[Source(url=facts.url, retrieved_at=facts.retrieved_at)],
        last_modified=facts.last_modified,
        trained_on=facts.trained_on,
        papers=facts.papers,
    )
