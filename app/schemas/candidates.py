"""Candidates proposed by scouts and the verdicts the code checker gives them."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Precision = Literal["fp32", "fp16", "bf16", "int8", "int4"]
ItemKind = Literal["model", "dataset", "method"]


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
    last_modified: datetime | None = None
    # Relationships recorded on the Hub (model card tags), used to connect the kit.
    trained_on: list[str] = Field(default_factory=list, description="dataset ids")
    papers: list[str] = Field(default_factory=list, description="arXiv ids")

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


class DatasetPick(BaseModel):
    """A verified dataset. Every field is read from the Hub or its dataset viewer."""

    repo_id: str
    licence: str
    downloads: int
    num_rows: int | None
    config: str | None = None
    splits: list[str]
    features: list[str]
    sample_rows: list[dict[str, str]] = Field(default_factory=list, max_length=3)
    languages: list[str] = Field(default_factory=list)
    last_modified: datetime | None = None
    score: float = Field(ge=0, le=1)
    why: str
    notes: list[str] = Field(default_factory=list)
    used_by: list[str] = Field(default_factory=list, description="chosen models trained on it")
    retrieved_at: datetime

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}"


class MethodCandidate(BaseModel):
    """A paper, guide or repo found by the method scout (unverified)."""

    title: str
    url: str
    kind: Literal["paper", "guide", "repo"]
    source: Literal["arxiv", "web", "hub-metadata"]
    snippet: str = ""
    arxiv_id: str | None = None
    published: str | None = None


class MethodPick(BaseModel):
    """A verified method resource: the link resolved (and the arXiv id exists, for papers)."""

    title: str
    url: str
    kind: Literal["paper", "guide", "repo"]
    source: Literal["arxiv", "web", "hub-metadata"]
    why: str
    arxiv_id: str | None = None
    published: str | None = None
    describes: list[str] = Field(default_factory=list, description="chosen models it describes")
    retrieved_at: datetime


class MethodSelection(BaseModel):
    """LLM output: which of the numbered method results to keep, best first."""

    picks: list[int] = Field(max_length=8, description="Indices from the numbered list")
    reasons: list[str] = Field(description="One short reason per pick, same order")


class RejectedCandidate(BaseModel):
    repo_id: str
    reasons: list[str]
    stage: Literal["existence", "constraints"]
    kind: ItemKind = "model"
