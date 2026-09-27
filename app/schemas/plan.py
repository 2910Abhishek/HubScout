"""The research plan the user approves before scouting starts."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_TAG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class ResearchPlan(BaseModel):
    hf_task: str = Field(
        description="Hugging Face pipeline tag for the task, e.g. 'automatic-speech-recognition', "
        "'text-classification', 'image-classification', 'text-generation'"
    )
    search_queries: list[str] = Field(
        min_length=1, max_length=4, description="Short Hub search queries (2-5 words each)"
    )
    steps: list[str] = Field(
        min_length=1, max_length=6, description="What HubScout will do, in plain language"
    )
    selection_criteria: list[str] = Field(
        default_factory=list, max_length=6, description="What makes a candidate good"
    )

    @field_validator("hf_task")
    @classmethod
    def _tag(cls, value: str) -> str:
        value = value.strip().lower().replace(" ", "-").replace("_", "-")
        if not _TAG.match(value):
            raise ValueError(f"not a Hub pipeline tag: {value!r}")
        return value


class PlanReview(BaseModel):
    """The human's answer at the plan-approval interrupt."""

    approved: bool
    feedback: str | None = None
