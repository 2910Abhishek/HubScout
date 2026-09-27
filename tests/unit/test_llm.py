"""Tier wiring: OpenRouter primary with Ollama fallback; judge pinned; no network."""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import load_settings
from app.llm import build_ollama_model, build_openrouter_model, get_llm, served_by
from app.rate_limit import RedisRateLimiter

FAKE_KEY = "sk-or-test-not-a-real-key"  # pragma: allowlist secret


class FailingChatModel(GenericFakeChatModel):
    """A chat model whose every call fails, like OpenRouter returning 429."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("429 Too Many Requests")


def fake(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


@pytest.fixture
def settings_with_key(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    return load_settings(env_files=None)


def test_primary_answers_when_healthy() -> None:
    llm = get_llm("strong", primary=fake("from openrouter"), fallback=fake("from ollama"))

    assert llm.invoke("hi").content == "from openrouter"


def test_falls_back_to_ollama_when_primary_fails() -> None:
    llm = get_llm(
        "cheap", primary=FailingChatModel(messages=iter([])), fallback=fake("from ollama")
    )

    assert llm.invoke("hi").content == "from ollama"


def test_judge_never_falls_back() -> None:
    llm = get_llm("judge", primary=FailingChatModel(messages=iter([])), fallback=fake("x"))

    with pytest.raises(RuntimeError, match="429"):
        llm.invoke("hi")


def test_without_openrouter_key_uses_ollama_only() -> None:
    llm = get_llm("strong", settings=load_settings(env_files=None), fallback=fake("local"))

    assert llm.invoke("hi").content == "local"


def test_judge_without_key_is_an_error() -> None:
    with pytest.raises(ValueError, match="judge"):
        get_llm("judge", settings=load_settings(env_files=None))


def test_tools_and_schema_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either"):
        get_llm("strong", tools=[{"type": "function"}], schema=ChatOpenAI, primary=fake("x"))  # type: ignore[arg-type]


def test_openrouter_model_is_configured_from_settings(settings_with_key: Any) -> None:
    limiter = RedisRateLimiter("redis://unused:1", rpm=20, rpd=50)
    model = build_openrouter_model("cheap", settings_with_key, rate_limiter=limiter)

    assert model.model_name == settings_with_key.llm.model_cheap
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == FAKE_KEY
    assert model.max_retries == 1
    assert model.rate_limiter is limiter
    assert FAKE_KEY not in repr(model)


def test_ollama_model_uses_v1_endpoint_and_no_retries() -> None:
    model = build_ollama_model(load_settings(env_files=None))

    assert model.openai_api_base == "http://localhost:11434/v1"
    assert model.max_retries == 0
    assert model.rate_limiter is None


def test_openrouter_model_requires_key() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_openrouter_model("strong", load_settings(env_files=None))


def test_served_by_reads_response_metadata() -> None:
    assert served_by(AIMessage(content="", response_metadata={"model_name": "a/b"})) == "a/b"
    assert served_by(AIMessage(content="")) == "unknown"
