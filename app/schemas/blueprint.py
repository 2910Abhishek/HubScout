"""HubScout's final output (proof-of-concept subset of card §10)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.schemas.candidates import OpenWeightCandidate, RejectedCandidate, Source
from app.schemas.constraints import Constraints


class AggregatorNarrative(BaseModel):
    """The only part of the blueprint the LLM writes; every fact it may use is given to it."""

    recommendation_summary: str = Field(description="3-5 sentences for the user")
    pick_reasons: list[str] = Field(description="Why the top pick wins, citing the given facts")
    risks: list[str] = Field(description="Concrete risks and caveats")


class Blueprint(BaseModel):
    constraints: Constraints
    open_weight_pick: OpenWeightCandidate | None
    open_weight_alternatives: list[OpenWeightCandidate] = Field(default_factory=list)
    rejected: list[RejectedCandidate] = Field(default_factory=list)
    recommendation_summary: str
    risks: list[str] = Field(default_factory=list)
    licensing_notes: list[str] = Field(default_factory=list)
    pipeline_mermaid: str | None = None
    sources: list[Source] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    created_at: datetime

    @model_validator(mode="after")
    def _mode_consistency(self) -> Blueprint:
        mode = self.constraints.deployment_mode
        if mode == "api" and (self.open_weight_pick or self.open_weight_alternatives):
            raise ValueError("api-mode blueprint must not recommend open-weight models")
        if mode in ("open_weight", "compare") and self.open_weight_pick is None and not self.notes:
            raise ValueError("no open-weight pick: notes must explain why")
        if self.open_weight_pick and self.open_weight_pick in self.open_weight_alternatives:
            raise ValueError("the pick must not also be listed as an alternative")
        return self
