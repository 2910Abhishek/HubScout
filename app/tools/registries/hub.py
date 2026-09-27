"""Facts about Hugging Face models, straight from the Hub API (the source of truth).

The checker never trusts what an LLM says about a model; it asks the Hub via this client.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HFValidationError, RepositoryNotFoundError

from app.config import Settings

_EXPAND: list[Any] = ["safetensors", "cardData", "pipeline_tag", "gated", "tags", "downloads"]


@dataclass(frozen=True)
class ModelFacts:
    repo_id: str
    exists: bool
    retrieved_at: datetime
    licence: str | None = None
    gated: bool = False
    pipeline_tag: str | None = None
    params: int | None = None
    dtypes: dict[str, int] = field(default_factory=dict)
    languages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    downloads: int = 0

    @property
    def requires_remote_code(self) -> bool:
        return "custom_code" in self.tags

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


class HubLookup(Protocol):
    """Anything that can return ModelFacts (real Hub client, or a fake in tests)."""

    async def model_facts(self, repo_id: str) -> ModelFacts: ...


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    return [str(v).lower() for v in value]


class HubClient:
    def __init__(self, settings: Settings) -> None:
        token = settings.data.hf_token
        self._api = HfApi(token=token.get_secret_value() if token else None)
        self._timeout = settings.data.http_timeout_s

    def _fetch(self, repo_id: str) -> ModelFacts:
        now = datetime.now(UTC)
        try:
            info = self._api.model_info(repo_id, expand=_EXPAND, timeout=self._timeout)
        except (RepositoryNotFoundError, HFValidationError):
            return ModelFacts(repo_id=repo_id, exists=False, retrieved_at=now)
        except GatedRepoError:
            return ModelFacts(repo_id=repo_id, exists=True, gated=True, retrieved_at=now)
        # to_dict() is untyped upstream.
        card: dict[str, Any] = info.card_data.to_dict() if info.card_data else {}  # type: ignore[no-untyped-call]
        licences = _as_list(card.get("license"))
        st = info.safetensors
        return ModelFacts(
            repo_id=info.id or repo_id,
            exists=True,
            retrieved_at=now,
            licence=licences[0] if licences else None,
            gated=bool(info.gated),
            pipeline_tag=info.pipeline_tag,
            params=st.total if st else None,
            dtypes=dict(st.parameters) if st else {},
            languages=_as_list(card.get("language")),
            tags=list(info.tags or []),
            downloads=info.downloads or 0,
        )

    async def model_facts(self, repo_id: str) -> ModelFacts:
        # huggingface_hub is synchronous; keep the event loop free.
        return await asyncio.to_thread(self._fetch, repo_id)
