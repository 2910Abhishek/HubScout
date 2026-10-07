"""The user's requirements, as understood by the Clarifier."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

TaskFamily = Literal["text", "speech", "vision"]


class ConstraintDraft(BaseModel):
    """What the LLM could extract from the conversation so far. Unknown fields stay None.

    Code (not the LLM) decides which of the missing fields must be asked about.
    """

    task_family: TaskFamily | None = Field(
        default=None, description="text, speech or vision; None if unclear"
    )
    task_description: str | None = Field(
        default=None, description="One-sentence restatement of the ML task"
    )
    commercial_use: bool | None = Field(default=None, description="None unless stated")
    languages: list[str] = Field(
        default_factory=list, description="ISO 639-1 codes, e.g. ['hi', 'en']"
    )
    gpu_vram_gb: float | None = Field(default=None, ge=0, description="GPU memory in GB")
    cpu_only: bool | None = Field(default=None, description="True if no GPU is available")
    needs_finetuning: bool | None = None
    licence_policy: list[str] = Field(
        default_factory=list, description="Allowed licence ids if the user named any"
    )
    monthly_volume: str | None = Field(default=None, description="e.g. '20000 audio minutes'")
    monthly_budget_usd: float | None = Field(default=None, ge=0)
    data_can_leave_org: bool | None = None
    max_latency_ms: int | None = Field(default=None, gt=0)

    @field_validator("languages", "licence_policy")
    @classmethod
    def _lower(cls, values: list[str]) -> list[str]:
        return sorted({v.strip().lower() for v in values if v.strip()})


class Constraints(BaseModel):
    """Settled requirements that every later stage relies on."""

    task_family: TaskFamily
    task_description: str
    commercial_use: bool
    languages: list[str] = Field(default_factory=list)
    gpu_vram_gb: float | None = Field(default=None, ge=0)
    cpu_only: bool = False
    needs_finetuning: bool = False
    licence_policy: list[str] = Field(default_factory=list)
    monthly_volume: str | None = None
    monthly_budget_usd: float | None = Field(default=None, ge=0)
    data_can_leave_org: bool | None = None
    max_latency_ms: int | None = Field(default=None, gt=0)
