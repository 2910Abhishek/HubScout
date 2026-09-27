"""Application configuration.

All tunable values in HubScout live in this module, in three layers:

1. Defaults below: every non-secret setting (model IDs, URLs, limits, policy).
2. `.env.infra` (auto-generated, git-ignored): local infrastructure secrets such as the
   Postgres password and Langfuse keys. Created by `scripts/init_env.sh`; nobody edits it.
3. `.env` (git-ignored): the user's external API keys only. See `.env.example`.

Any setting can still be overridden by an environment variable of the same name (plus the
group's prefix), which is how tests and CI configure it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILES: tuple[Path, ...] = (PROJECT_ROOT / ".env.infra", PROJECT_ROOT / ".env")

DeploymentModeSetting = Literal["ask", "api", "open_weight", "compare"]
LlmTier = Literal["strong", "cheap", "judge"]


class MissingSecretError(RuntimeError):
    """A secret needed for the requested operation is not configured."""


class _Group(BaseSettings):
    """Base for all settings groups: shared env files, unknown keys ignored, immutable."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )


class OpenRouterSettings(_Group):
    """OpenRouter: primary LLM provider for HubScout's own agents, plus the model catalogue."""

    model_config = SettingsConfigDict(env_prefix="OPENROUTER_")

    api_key: SecretStr | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    models_url: str = "https://openrouter.ai/api/v1/models"
    # Free-model limits: 20/min always; 50/day without purchased credits (1000/day with >= $10).
    rpm: int = Field(default=20, gt=0)
    rpd: int = Field(default=50, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    # Retries spend the daily quota, so keep them low; the Ollama fallback covers failures.
    max_retries: int = Field(default=1, ge=0)
    app_title: str = "HubScout"


class LlmSettings(_Group):
    """Model ID per tier, chosen from the live OpenRouter catalogue.

    Verified on 2026-09-27 to exist and support `tools` + `structured_outputs`. The free
    lineup changes often; the integration smoke test re-checks these IDs.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_")

    model_strong: str = "nvidia/nemotron-3-super-120b-a12b:free"
    model_cheap: str = "qwen/qwen3.8-27b:free"
    # Pinned, never falls back (evaluation judge).
    model_judge: str = "nvidia/nemotron-3-super-120b-a12b:free"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_concurrency: int = Field(default=3, gt=0)

    def model_for(self, tier: LlmTier) -> str:
        return {"strong": self.model_strong, "cheap": self.model_cheap, "judge": self.model_judge}[
            tier
        ]


class OllamaSettings(_Group):
    """Local Ollama fallback, reached through its OpenAI-compatible `/v1` endpoint."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"
    # Small tool-calling model that fits this machine (CPU, 22 GB RAM); no "thinking" tokens.
    model: str = "qwen3:4b-instruct"
    # Ollama ignores the key, but the OpenAI client requires a non-empty value.
    api_key: SecretStr = SecretStr("ollama")  # pragma: allowlist secret
    timeout_s: float = Field(default=180.0, gt=0)

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1"


class DataSourceSettings(_Group):
    """Registries, catalogues and paper sources used by the scouts."""

    hf_token: SecretStr | None = None
    hf_mcp_url: str = "https://huggingface.co/mcp"
    artificial_analysis_api_key: SecretStr | None = None
    artificial_analysis_base_url: str = "https://artificialanalysis.ai/api/v2"
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"
    pypi_base_url: str = "https://pypi.org/pypi"
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    http_timeout_s: float = Field(default=30.0, gt=0)


class SearchSettings(_Group):
    """Web search for the web scout. Results are leads, never truth.

    Tavily is the primary backend (agent-oriented results with extracted page content);
    self-hosted SearXNG is the free, unlimited fallback when the Tavily key is missing or its
    monthly credits (free tier: 1,000; basic search = 1, advanced = 2) run out.
    """

    search_backend: Literal["tavily", "searxng"] = "tavily"
    tavily_api_key: SecretStr | None = None
    tavily_base_url: str = "https://api.tavily.com"
    tavily_search_depth: Literal["basic", "advanced"] = "basic"
    tavily_max_results: int = Field(default=5, gt=0, le=20)
    searxng_url: str = "http://localhost:8888"


class InfraSettings(_Group):
    """Postgres (checkpointer, store, pgvector) and Redis (cache, rate limiting)."""

    postgres_user: str = "hubscout"
    postgres_password: SecretStr | None = None
    postgres_db: str = "hubscout"
    postgres_host: str = "localhost"
    # Host port 5433: 5432 is already used by a Postgres installed on the host.
    postgres_port: int = Field(default=5433, gt=0, lt=65536)
    redis_url: str = "redis://localhost:6379/0"
    registry_cache_ttl_s: int = Field(default=6 * 3600, gt=0)
    pricing_cache_ttl_s: int = Field(default=24 * 3600, gt=0)
    search_cache_ttl_s: int = Field(default=3600, gt=0)

    @property
    def database_url(self) -> SecretStr:
        """Connection string for the HubScout Postgres, built from the parts above."""
        if self.postgres_password is None:
            raise MissingSecretError(
                "POSTGRES_PASSWORD is not set; run scripts/init_env.sh to generate .env.infra"
            )
        password = self.postgres_password.get_secret_value()
        return SecretStr(
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class SandboxSettings(_Group):
    """Hardened Docker sandbox used by the verifier (Phase 4)."""

    model_config = SettingsConfigDict(env_prefix="SANDBOX_")

    image: str = "hubscout-sandbox:latest"
    timeout_s: int = Field(default=600, gt=0)
    cpus: float = Field(default=2.0, gt=0)
    memory_mb: int = Field(default=4096, gt=0)
    disk_mb: int = Field(default=10240, gt=0)
    seed: int = 42
    sample_size: int = Field(default=100, gt=0)
    api_verify_max_cost_usd: float = Field(default=1.0, ge=0)


class VerifierSettings(_Group):
    """Google ADK verifier served over A2A (Phase 4)."""

    verifier_a2a_url: str = "http://localhost:9000"


class ObservabilitySettings(_Group):
    """Langfuse tracing, LangSmith (Studio), and logging."""

    langfuse_host: str = "http://localhost:3000"
    # Generated into .env.infra together with the local Langfuse project.
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class PolicySettings(_Group):
    """Recommendation policy defaults."""

    licence_allowlist: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"]
    )
    default_deployment_mode: DeploymentModeSetting = "ask"
    max_critic_loops: int = Field(default=2, ge=0, le=5)
    scout_concurrency: int = Field(default=3, gt=0)
    # Clarify/plan loops: how many times the graph may re-ask before using defaults.
    max_clarify_rounds: int = Field(default=2, ge=0, le=5)
    max_plan_revisions: int = Field(default=2, ge=0, le=5)
    # Open-weight scout: tool-calling rounds with the HF MCP server, and candidates kept.
    scout_max_tool_rounds: int = Field(default=4, gt=0, le=10)
    max_candidates: int = Field(default=5, gt=0, le=10)
    # VRAM estimate = params x bytes-per-param x overhead (activations, KV cache, runtime).
    vram_overhead_factor: float = Field(default=1.2, ge=1.0)
    # CPU-only users: largest model footprint (GB) considered practical in system RAM.
    cpu_max_model_gb: float = Field(default=16.0, gt=0)

    @field_validator("licence_allowlist", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value


class ApiSettings(_Group):
    """FastAPI backend (Phase 6)."""

    model_config = SettingsConfigDict(env_prefix="API_")

    # Generated into .env.infra.
    auth_key: SecretStr | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)


class Settings(BaseModel):
    """All HubScout settings. Build with `load_settings()` or the cached `get_settings()`."""

    model_config = ConfigDict(frozen=True)

    openrouter: OpenRouterSettings
    llm: LlmSettings
    ollama: OllamaSettings
    data: DataSourceSettings
    search: SearchSettings
    infra: InfraSettings
    sandbox: SandboxSettings
    verifier: VerifierSettings
    observability: ObservabilitySettings
    policy: PolicySettings
    api: ApiSettings


def load_settings(env_files: tuple[Path, ...] | Literal["default"] | None = "default") -> Settings:
    """Load settings from defaults, env files, and the process environment (highest priority).

    `"default"` resolves `DEFAULT_ENV_FILES` at call time (so tests can redirect it);
    `None` skips env files and reads the process environment only.
    """
    resolved = DEFAULT_ENV_FILES if env_files == "default" else env_files
    kwargs: dict[str, Any] = {"_env_file": resolved}
    return Settings(
        openrouter=OpenRouterSettings(**kwargs),
        llm=LlmSettings(**kwargs),
        ollama=OllamaSettings(**kwargs),
        data=DataSourceSettings(**kwargs),
        search=SearchSettings(**kwargs),
        infra=InfraSettings(**kwargs),
        sandbox=SandboxSettings(**kwargs),
        verifier=VerifierSettings(**kwargs),
        observability=ObservabilitySettings(**kwargs),
        policy=PolicySettings(**kwargs),
        api=ApiSettings(**kwargs),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests call `get_settings.cache_clear()` after changing env."""
    return load_settings()
