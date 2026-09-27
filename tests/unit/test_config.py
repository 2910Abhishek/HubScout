"""Settings load from defaults, env files and the environment; validate; never leak secrets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import MissingSecretError, get_settings, load_settings

FAKE_OPENROUTER_KEY = "sk-or-test-not-a-real-key"  # pragma: allowlist secret
FAKE_PG_PASSWORD = "fake-pg-password"  # pragma: allowlist secret


def test_works_with_no_env_at_all() -> None:
    settings = load_settings(env_files=None)

    assert settings.openrouter.api_key is None
    assert settings.openrouter.rpm == 20
    assert settings.openrouter.rpd == 50
    assert settings.llm.model_strong
    assert settings.ollama.openai_base_url == "http://localhost:11434/v1"
    assert settings.policy.default_deployment_mode == "ask"
    assert settings.search.search_backend == "tavily"
    assert settings.search.tavily_api_key is None
    assert settings.policy.licence_allowlist == [
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
    ]


def test_model_for_maps_each_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL_STRONG", "vendor/strong:free")
    monkeypatch.setenv("LLM_MODEL_CHEAP", "vendor/cheap:free")
    monkeypatch.setenv("LLM_MODEL_JUDGE", "vendor/judge:free")
    llm = load_settings(env_files=None).llm

    assert llm.model_for("strong") == "vendor/strong:free"
    assert llm.model_for("cheap") == "vendor/cheap:free"
    assert llm.model_for("judge") == "vendor/judge:free"


def test_env_overrides_and_csv_list_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_RPD", "1000")
    monkeypatch.setenv("LICENCE_ALLOWLIST", "Apache-2.0, MIT ,,bsd-3-clause")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434/")

    settings = load_settings(env_files=None)

    assert settings.openrouter.rpd == 1000
    assert settings.policy.licence_allowlist == ["apache-2.0", "mit", "bsd-3-clause"]
    assert settings.ollama.openai_base_url == "http://ollama.internal:11434/v1"


def test_empty_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # A freshly copied .env has keys with empty values; they must mean "not set".
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "")

    settings = load_settings(env_files=None)

    assert settings.openrouter.api_key is None
    assert settings.openrouter.base_url == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENROUTER_RPM", "0"),
        ("DEFAULT_DEPLOYMENT_MODE", "on_prem"),
        ("MAX_CRITIC_LOOPS", "99"),
        ("API_PORT", "70000"),
        ("SEARCH_BACKEND", "bing"),
        ("TAVILY_MAX_RESULTS", "50"),
    ],
)
def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        load_settings(env_files=None)


def test_database_url_built_from_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", FAKE_PG_PASSWORD)

    url = load_settings(env_files=None).infra.database_url

    assert url.get_secret_value() == (
        f"postgresql://hubscout:{FAKE_PG_PASSWORD}@localhost:5433/hubscout"
    )
    assert FAKE_PG_PASSWORD not in repr(url)


def test_database_url_without_password_raises() -> None:
    infra = load_settings(env_files=None).infra

    with pytest.raises(MissingSecretError, match="POSTGRES_PASSWORD"):
        _ = infra.database_url


def test_secrets_never_appear_in_repr_or_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER_KEY)
    monkeypatch.setenv("POSTGRES_PASSWORD", FAKE_PG_PASSWORD)

    settings = load_settings(env_files=None)
    rendered = repr(settings) + str(settings) + settings.model_dump_json()

    assert FAKE_OPENROUTER_KEY not in rendered
    assert FAKE_PG_PASSWORD not in rendered
    assert settings.openrouter.api_key is not None
    assert settings.openrouter.api_key.get_secret_value() == FAKE_OPENROUTER_KEY


def test_env_files_layered_and_environment_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    infra = tmp_path / ".env.infra"
    infra.write_text("POSTGRES_PASSWORD=from-infra\nOPENROUTER_RPM=5\n")
    keys = tmp_path / ".env"
    keys.write_text("OPENROUTER_API_KEY=from-dotenv\nOPENROUTER_RPM=7\n")
    monkeypatch.setenv("OPENROUTER_RPD", "9")

    settings = load_settings(env_files=(infra, keys))

    assert settings.infra.postgres_password is not None
    assert settings.infra.postgres_password.get_secret_value() == "from-infra"
    assert settings.openrouter.api_key is not None
    assert settings.openrouter.api_key.get_secret_value() == "from-dotenv"
    assert settings.openrouter.rpm == 7  # later file wins
    assert settings.openrouter.rpd == 9  # process environment wins over files


def test_default_env_files_resolved_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SEARXNG_URL=http://redirected:8080\n")
    monkeypatch.setattr("app.config.DEFAULT_ENV_FILES", (env_file,))

    assert load_settings().search.searxng_url == "http://redirected:8080"


def test_settings_are_immutable() -> None:
    settings = load_settings(env_files=None)

    with pytest.raises(ValidationError):
        settings.openrouter.rpm = 999  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
