"""HubScout's output: the starter kit (validated), rendered to README by code."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.schemas.candidates import DatasetPick, MethodPick, OpenWeightCandidate, RejectedCandidate
from app.schemas.constraints import Constraints

MAX_PER_SECTION = 5
MAX_LINKS = 15


class KitNarrative(BaseModel):
    """The only LLM-written prose in the kit; it may use only the facts it is given."""

    tldr: str = Field(description="3-4 sentences: the recommended combination and why")
    fit_story: str = Field(
        description="One sentence on how the top dataset, method and model fit together"
    )


class StarterKit(BaseModel):
    constraints: Constraints
    tldr: str
    fit_story: str
    models: list[OpenWeightCandidate] = Field(default_factory=list, max_length=MAX_PER_SECTION)
    datasets: list[DatasetPick] = Field(default_factory=list, max_length=MAX_PER_SECTION)
    methods: list[MethodPick] = Field(default_factory=list, max_length=MAX_PER_SECTION)
    rejected: list[RejectedCandidate] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    quick_start: str | None = None
    models_used: list[str] = Field(default_factory=list)
    sources_checked: int = 0
    created_at: datetime

    @property
    def links(self) -> list[str]:
        return (
            [m.url for m in self.models]
            + [d.url for d in self.datasets]
            + [m.url for m in self.methods]
        )

    @model_validator(mode="after")
    def _limits(self) -> StarterKit:
        links = self.links
        if len(links) > MAX_LINKS:
            raise ValueError(f"at most {MAX_LINKS} links, got {len(links)}")
        if len(set(links)) != len(links):
            raise ValueError("duplicate links in the kit")
        if not links and not self.gaps:
            raise ValueError("an empty kit must explain the gaps")
        return self
