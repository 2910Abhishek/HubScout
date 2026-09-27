"""Application configuration, loaded from environment variables and `.env`.

Every tunable value in HubScout (URLs, keys, model IDs, limits) lives here. Settings are
grouped by concern; each group reads its own env-var prefix. `.env.example` documents every
variable. Public, stable endpoints have defaults; anything secret or environment-specific
(keys, model IDs, database URL) must come from the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

DeploymentModeSetting = Literal["ask", "api", "open_weight", "compare"]
LlmTier = Literal["strong", "cheap", "judge"]


class _Group(BaseSettings):
    """Base for all settings groups: shared `.env` handling, unknown keys ignored."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
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
    # Optional attribution headers (HTTP-Referer / X-Title) shown on openrouter.ai.
    app_url: str | None = None
    app_title: str = "HubScout"


class LlmSettings(_Group):
    """Model IDs per tier. Chosen from the live catalogue; never hardcoded in code."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    model_strong: str
    model_cheap: str
    model_judge: str
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
    model: str
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
    semantic_scholar_api_key: SecretStr | None = None
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"
    pypi_base_url: str = "https://pypi.org/pypi"
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    http_timeout_s: float = Field(default=30.0, gt=0)


class SearchSettings(_Group):
    """Self-hosted SearXNG web search (results are leads, never truth)."""

    searxng_url: str = "http://localhost:8888"


class InfraSettings(_Group):
    """Postgres (checkpointer, store, pgvector) and Redis (cache, rate limiting)."""

    database_url: SecretStr
    redis_url: str = "redis://localhost:6379/0"
    registry_cache_ttl_s: int = Field(default=6 * 3600, gt=0)
    pricing_cache_ttl_s: int = Field(default=24 * 3600, gt=0)
    search_cache_ttl_s: int = Field(default=3600, gt=0)


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

    @field_validator("licence_allowlist", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value


class ApiSettings(_Group):
    """FastAPI backend (Phase 6)."""

    model_config = SettingsConfigDict(env_prefix="API_")

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


def load_settings(env_file: Path | Literal["default"] | None = "default") -> Settings:
    """Load settings from the process environment plus an env file.

    `"default"` resolves `DEFAULT_ENV_FILE` at call time (so tests can redirect it);
    `None` reads the process environment only.
    """
    resolved = DEFAULT_ENV_FILE if env_file == "default" else env_file
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
