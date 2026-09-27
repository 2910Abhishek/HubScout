"""Settings load from the environment, validate, and never leak secrets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import get_settings, load_settings

REQUIRED_ENV = {
    "LLM_MODEL_STRONG": "vendor/strong-model:free",
    "LLM_MODEL_CHEAP": "vendor/cheap-model:free",
    "LLM_MODEL_JUDGE": "vendor/judge-model:free",
    "OLLAMA_MODEL": "local-model:4b",
    "DATABASE_URL": "postgresql://u:fake-pw@localhost:5433/db",  # pragma: allowlist secret
}
FAKE_OPENROUTER_KEY = "sk-or-test-not-a-real-key"


@pytest.fixture
def required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.mark.usefixtures("required_env")
def test_defaults_applied_when_only_required_vars_set() -> None:
    settings = load_settings(env_file=None)

    assert settings.openrouter.rpm == 20
    assert settings.openrouter.rpd == 50
    assert settings.openrouter.api_key is None
    assert settings.ollama.openai_base_url == "http://localhost:11434/v1"
    assert settings.policy.default_deployment_mode == "ask"
    assert settings.policy.licence_allowlist == [
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
    ]


@pytest.mark.usefixtures("required_env")
def test_model_for_maps_each_tier() -> None:
    llm = load_settings(env_file=None).llm

    assert llm.model_for("strong") == REQUIRED_ENV["LLM_MODEL_STRONG"]
    assert llm.model_for("cheap") == REQUIRED_ENV["LLM_MODEL_CHEAP"]
    assert llm.model_for("judge") == REQUIRED_ENV["LLM_MODEL_JUDGE"]


@pytest.mark.usefixtures("required_env")
def test_env_overrides_and_csv_list_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_RPD", "1000")
    monkeypatch.setenv("LICENCE_ALLOWLIST", "Apache-2.0, MIT ,,bsd-3-clause")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434/")

    settings = load_settings(env_file=None)

    assert settings.openrouter.rpd == 1000
    assert settings.policy.licence_allowlist == ["apache-2.0", "mit", "bsd-3-clause"]
    assert settings.ollama.openai_base_url == "http://ollama.internal:11434/v1"


@pytest.mark.usefixtures("required_env")
def test_empty_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # .env.example ships keys with empty values; they must not override defaults.
    monkeypatch.setenv("OPENROUTER_BASE_URL", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    settings = load_settings(env_file=None)

    assert settings.openrouter.base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter.api_key is None


def test_missing_required_values_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in REQUIRED_ENV:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError, match="model_strong"):
        load_settings(env_file=None)


@pytest.mark.usefixtures("required_env")
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENROUTER_RPM", "0"),
        ("DEFAULT_DEPLOYMENT_MODE", "on_prem"),
        ("MAX_CRITIC_LOOPS", "99"),
        ("API_PORT", "70000"),
    ],
)
def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        load_settings(env_file=None)


@pytest.mark.usefixtures("required_env")
def test_secrets_never_appear_in_repr_or_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER_KEY)

    settings = load_settings(env_file=None)
    rendered = repr(settings) + str(settings) + settings.model_dump_json()

    assert FAKE_OPENROUTER_KEY not in rendered
    assert "fake-pw" not in rendered
    assert settings.openrouter.api_key is not None
    assert settings.openrouter.api_key.get_secret_value() == FAKE_OPENROUTER_KEY


@pytest.mark.usefixtures("required_env")
def test_env_file_is_read_and_environment_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_RPM=7\nSEARXNG_URL=http://from-file:8080\n")
    monkeypatch.setenv("OPENROUTER_RPM", "9")

    settings = load_settings(env_file=env_file)

    assert settings.openrouter.rpm == 9
    assert settings.search.searxng_url == "http://from-file:8080"


@pytest.mark.usefixtures("required_env")
def test_settings_are_immutable() -> None:
    settings = load_settings(env_file=None)

    with pytest.raises(ValidationError):
        settings.openrouter.rpm = 999  # type: ignore[misc]


@pytest.mark.usefixtures("required_env")
def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_default_env_file_is_resolved_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    env_file = tmp_path / "custom.env"
    env_file.write_text("SEARXNG_URL=http://redirected:8080\n")
    monkeypatch.setattr("app.config.DEFAULT_ENV_FILE", env_file)

    assert load_settings().search.searxng_url == "http://redirected:8080"
