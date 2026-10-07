"""Programmatic checks for dataset candidates. No LLM: pure, testable functions."""

from __future__ import annotations

from app.config import PolicySettings
from app.schemas import Constraints, DatasetPick, RejectedCandidate
from app.tools.checks.languages import lacks_key_language, missing_languages
from app.tools.checks.open_weight import allowed_licences, popularity
from app.tools.registries.hub import DatasetFacts

# Licences that clearly permit commercial use of a dataset (on top of the model allowlist).
PERMISSIVE_DATA_LICENCES = frozenset(
    {"cc0-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc-by-sa-3.0", "odc-by", "pddl", "odbl"}
)
# Modality hints: a speech dataset should have an audio column, a vision one an image column.
_MODALITY_FEATURE = {"speech": "(Audio)", "vision": "(Image)"}


def check_dataset(
    facts: DatasetFacts,
    constraints: Constraints,
    hf_task: str | None,
    policy: PolicySettings,
    why: str,
) -> DatasetPick | RejectedCandidate:
    if not facts.exists:
        return RejectedCandidate(
            repo_id=facts.repo_id,
            stage="existence",
            kind="dataset",
            reasons=[
                f"could not verify: {facts.error}"
                if facts.error
                else "dataset does not exist on the Hub"
            ],
        )

    reasons: list[str] = []
    notes: list[str] = []
    if facts.gated:
        reasons.append("gated dataset: requires accepting terms / manual approval")

    allowed = allowed_licences(constraints, policy)
    if facts.licence is None:
        reasons.append("licence missing from the dataset card")
    elif allowed is not None and facts.licence not in allowed | PERMISSIVE_DATA_LICENCES:
        reasons.append(f"licence '{facts.licence}' not allowed for commercial use")

    if not facts.gated and not facts.viewer_ok:
        reasons.append("no working dataset viewer (may need custom loading code); cannot inspect")
    elif not facts.gated and not facts.splits:
        reasons.append("no splits reported")

    marker = _MODALITY_FEATURE.get(constraints.task_family)
    if marker and facts.features and not any(marker in f for f in facts.features):
        reasons.append(
            f"no {marker.strip('()').lower()} column for a {constraints.task_family} task"
        )

    if hf_task and facts.task_categories and hf_task not in facts.task_categories:
        notes.append(f"card lists tasks {facts.task_categories[:3]}, not '{hf_task}'")

    missing = missing_languages(constraints.languages, facts.languages)
    if lacks_key_language(constraints.languages, facts.languages):
        reasons.append(f"card lists none of the required languages {missing}")
    elif missing:
        notes.append(f"language support for {missing} not stated on the card")

    if reasons or facts.licence is None:
        return RejectedCandidate(
            repo_id=facts.repo_id, stage="constraints", kind="dataset", reasons=reasons
        )
    return DatasetPick(
        repo_id=facts.repo_id,
        licence=facts.licence,
        downloads=facts.downloads,
        num_rows=facts.num_rows,
        config=facts.config,
        splits=facts.splits,
        features=facts.features,
        sample_rows=facts.sample_rows,
        languages=facts.languages,
        last_modified=facts.last_modified,
        score=round(popularity(facts.downloads), 3),
        why=why,
        notes=notes,
        retrieved_at=facts.retrieved_at,
    )
