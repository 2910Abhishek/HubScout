"""Facts about Hugging Face models and datasets, straight from the Hub (the source of truth).

The checkers never trust what an LLM says about a repo; they ask the Hub via this client.
Dataset usability (splits, size, features, sample rows) comes from the Hub's dataset viewer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HFValidationError, RepositoryNotFoundError

from app.config import Settings

logger = logging.getLogger(__name__)
RETRY_DELAY_S = 1.0

_MODEL_EXPAND: list[Any] = [
    "safetensors",
    "cardData",
    "pipeline_tag",
    "gated",
    "tags",
    "downloads",
    "lastModified",
]
_DATASET_EXPAND: list[Any] = ["cardData", "gated", "tags", "downloads", "lastModified"]
# Full-model weight files, in order of preference (count one format only).
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_NOT_WEIGHTS = ("adapter_model", "optimizer", "training_args", "scheduler", "rng_state")
ASSUMED_BYTES_PER_PARAM = 2.0  # 16-bit weights; overestimates (safe side) for fp32 files
SAMPLE_ROWS = 3
MAX_CELL_CHARS = 120


def estimate_params_from_files(files: dict[str, int]) -> int | None:
    """Rough parameter count from weight-file sizes, for repos without safetensors metadata."""
    for suffix in _WEIGHT_SUFFIXES:
        sizes = [
            size
            for name, size in files.items()
            if name.endswith(suffix) and not any(bad in name for bad in _NOT_WEIGHTS)
        ]
        if sizes:
            return int(sum(sizes) / ASSUMED_BYTES_PER_PARAM)
    return None


def tag_values(tags: list[str], prefix: str) -> list[str]:
    """Values of `prefix:value` tags, e.g. tag_values(tags, "dataset") -> ["org/data"]."""
    start = f"{prefix}:"
    return [t[len(start) :] for t in tags if t.startswith(start)]


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
    params_estimated: bool = False
    adapter_only: bool = False
    last_modified: datetime | None = None
    error: str | None = None  # set when the Hub could not be reached (not verified)

    @property
    def requires_remote_code(self) -> bool:
        return "custom_code" in self.tags

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    @property
    def trained_on(self) -> list[str]:
        return tag_values(self.tags, "dataset")

    @property
    def papers(self) -> list[str]:
        return tag_values(self.tags, "arxiv")


@dataclass(frozen=True)
class DatasetFacts:
    repo_id: str
    exists: bool
    retrieved_at: datetime
    licence: str | None = None
    gated: bool = False
    tags: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    task_categories: list[str] = field(default_factory=list)
    downloads: int = 0
    last_modified: datetime | None = None
    viewer_ok: bool = False
    config: str | None = None
    num_rows: int | None = None
    splits: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None  # set when the Hub could not be reached (not verified)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}"


class HubLookup(Protocol):
    """Anything that can return Hub facts (the real client, or a fake in tests)."""

    async def model_facts(self, repo_id: str) -> ModelFacts: ...

    async def dataset_facts(self, repo_id: str) -> DatasetFacts: ...


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    return [str(v).lower() for v in value]


def _cell(value: Any, feature_type: str) -> str:
    """Readable sample cell: media becomes a placeholder, long text is cut."""
    if feature_type in ("Audio", "Image", "Video") or isinstance(value, (dict, list)):
        return f"<{feature_type.lower()}>" if feature_type else "<data>"
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


class HubClient:
    def __init__(self, settings: Settings) -> None:
        token = settings.data.hf_token
        self._token = token.get_secret_value() if token else None
        self._api = HfApi(token=self._token)
        self._timeout = settings.data.http_timeout_s
        self._viewer = settings.data.hf_datasets_server_url.rstrip("/")

    # ------------------------------------------------------------------ models
    def _fetch_model(self, repo_id: str) -> ModelFacts:
        now = datetime.now(UTC)
        try:
            info = self._api.model_info(repo_id, expand=_MODEL_EXPAND, timeout=self._timeout)
        except (RepositoryNotFoundError, HFValidationError):
            return ModelFacts(repo_id=repo_id, exists=False, retrieved_at=now)
        except GatedRepoError:
            return ModelFacts(repo_id=repo_id, exists=True, gated=True, retrieved_at=now)
        # to_dict() is untyped upstream.
        card: dict[str, Any] = info.card_data.to_dict() if info.card_data else {}  # type: ignore[no-untyped-call]
        licences = _as_list(card.get("license"))
        st = info.safetensors
        params = st.total if st else None
        estimated = adapter_only = False
        if params is None:
            files = self._file_sizes(repo_id)
            names = set(files)
            adapter_only = any(n.startswith("adapter_") for n in names) and not any(
                n.endswith(_WEIGHT_SUFFIXES) and not n.startswith("adapter_") for n in names
            )
            if not adapter_only:
                params = estimate_params_from_files(files)
                estimated = params is not None
        return ModelFacts(
            repo_id=info.id or repo_id,
            exists=True,
            retrieved_at=now,
            licence=licences[0] if licences else None,
            gated=bool(info.gated),
            pipeline_tag=info.pipeline_tag,
            params=params,
            params_estimated=estimated,
            adapter_only=adapter_only,
            dtypes=dict(st.parameters) if st else {},
            languages=_as_list(card.get("language")),
            tags=list(info.tags or []),
            downloads=info.downloads or 0,
            last_modified=info.last_modified,
        )

    def _file_sizes(self, repo_id: str) -> dict[str, int]:
        info = self._api.model_info(repo_id, files_metadata=True, timeout=self._timeout)
        return {s.rfilename: s.size or 0 for s in info.siblings or []}

    async def model_facts(self, repo_id: str) -> ModelFacts:
        # huggingface_hub is synchronous; keep the event loop free. One retry on network errors.
        for attempt in (1, 2):
            try:
                return await asyncio.to_thread(self._fetch_model, repo_id)
            except Exception as exc:
                logger.warning("Hub model lookup %s failed (try %d): %s", repo_id, attempt, exc)
                last = exc
                await asyncio.sleep(RETRY_DELAY_S)
        return ModelFacts(
            repo_id=repo_id,
            exists=False,
            retrieved_at=datetime.now(UTC),
            error=f"Hub unreachable ({type(last).__name__})",
        )

    # ---------------------------------------------------------------- datasets
    def _viewer_get(self, client: httpx.Client, path: str, **params: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        resp = client.get(f"{self._viewer}/{path}", params=params, headers=headers)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def _fetch_dataset(self, repo_id: str) -> DatasetFacts:
        now = datetime.now(UTC)
        try:
            info = self._api.dataset_info(repo_id, expand=_DATASET_EXPAND, timeout=self._timeout)
        except (RepositoryNotFoundError, HFValidationError):
            return DatasetFacts(repo_id=repo_id, exists=False, retrieved_at=now)
        except GatedRepoError:
            return DatasetFacts(repo_id=repo_id, exists=True, gated=True, retrieved_at=now)
        card: dict[str, Any] = info.card_data.to_dict() if info.card_data else {}  # type: ignore[no-untyped-call]
        tags = list(info.tags or [])
        licences = _as_list(card.get("license")) or tag_values(tags, "license")
        base = DatasetFacts(
            repo_id=info.id or repo_id,
            exists=True,
            retrieved_at=now,
            licence=licences[0] if licences else None,
            gated=bool(info.gated),
            tags=tags,
            languages=_as_list(card.get("language")) or tag_values(tags, "language"),
            task_categories=_as_list(card.get("task_categories"))
            or tag_values(tags, "task_categories"),
            downloads=info.downloads or 0,
            last_modified=info.last_modified,
        )
        if base.gated:
            return base
        with httpx.Client(timeout=self._timeout) as client:
            valid = self._viewer_get(client, "is-valid", dataset=base.repo_id)
            if not valid.get("preview"):
                return base
            split_list = self._viewer_get(client, "splits", dataset=base.repo_id).get("splits", [])
            if not split_list:
                return base
            first = split_list[0]
            config = str(first.get("config"))
            splits = [str(s["split"]) for s in split_list if s.get("config") == config]
            size = self._viewer_get(client, "size", dataset=base.repo_id).get("size", {})
            rows = self._viewer_get(
                client, "first-rows", dataset=base.repo_id, config=config, split=splits[0]
            )
        feature_types = {
            str(f.get("name")): str((f.get("type") or {}).get("_type", ""))
            for f in rows.get("features", [])
        }
        samples = [
            {name: _cell(r.get("row", {}).get(name), typ) for name, typ in feature_types.items()}
            for r in rows.get("rows", [])[:SAMPLE_ROWS]
        ]
        return DatasetFacts(
            **{
                **base.__dict__,
                "viewer_ok": True,
                "config": None if config == "default" else config,
                "num_rows": (size.get("dataset") or {}).get("num_rows"),
                "splits": splits,
                "features": [f"{n} ({t})" if t else n for n, t in feature_types.items()],
                "sample_rows": samples,
            }
        )

    async def dataset_facts(self, repo_id: str) -> DatasetFacts:
        for attempt in (1, 2):
            try:
                return await asyncio.to_thread(self._fetch_dataset, repo_id)
            except Exception as exc:
                logger.warning("Hub dataset lookup %s failed (try %d): %s", repo_id, attempt, exc)
                last = exc
                await asyncio.sleep(RETRY_DELAY_S)
        return DatasetFacts(
            repo_id=repo_id,
            exists=False,
            retrieved_at=datetime.now(UTC),
            error=f"Hub unreachable ({type(last).__name__})",
        )
