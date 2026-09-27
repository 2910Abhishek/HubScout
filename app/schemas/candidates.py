"""Candidates proposed by scouts and the verdicts the code checker gives them."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Precision = Literal["fp32", "fp16", "bf16", "int8", "int4"]


class Source(BaseModel):
    # Plain str (validated) rather than HttpUrl: LangGraph's checkpoint serializer
    # cannot encode pydantic's Url type.
    url: str
    retrieved_at: datetime

    @field_validator("url")
    @classmethod
    def _http(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("source url must be http(s)")
        return value


class ScoutCandidate(BaseModel):
    """A model the scout LLM proposes. Nothing here is trusted until the checker verifies it."""

    repo_id: str = Field(description="Exact Hugging Face repo id, e.g. 'openai/whisper-small'")
    why: str = Field(description="One sentence on why it fits")


class ScoutReport(BaseModel):
    candidates: list[ScoutCandidate] = Field(min_length=0, max_length=8)


class OpenWeightCandidate(BaseModel):
    """A verified open-weight model (card §10.2). Numbers come from code, not the LLM."""

    repo_id: str
    licence: str
    params_billion: float
    est_vram_gb: float
    precision: Precision
    pipeline_tag: str | None
    downloads: int
    fit_score: float = Field(ge=0, le=1)
    reasons: list[str]
    notes: list[str] = Field(default_factory=list)
    sources: list[Source]


class RejectedCandidate(BaseModel):
    repo_id: str
    reasons: list[str]
    stage: Literal["existence", "constraints"]
