"""Tiered chat models for HubScout's own agents.

`get_llm(tier)` returns a runnable that calls OpenRouter (rate limited through Redis) and
falls back to the local Ollama model on any error: HTTP failure, 429, daily quota spent,
or a structured-output validation failure. The `judge` tier never falls back, so evaluation
results always come from the same pinned model.

Tools and structured-output schemas are applied to the primary AND the fallback model before
they are combined, so both branches accept the same input and return the same shape.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import LlmTier, Settings, get_settings
from app.rate_limit import RedisRateLimiter

logger = logging.getLogger(__name__)

StructuredMethod = Literal["function_calling", "json_schema", "json_mode"]
ToolLike = BaseTool | type[BaseModel] | dict[str, Any]


@lru_cache(maxsize=1)
def _shared_rate_limiter(redis_url: str, rpm: int, rpd: int) -> RedisRateLimiter:
    """One limiter per process; the counters themselves are shared through Redis."""
    return RedisRateLimiter(redis_url, rpm=rpm, rpd=rpd)


def build_openrouter_model(
    tier: LlmTier,
    settings: Settings,
    *,
    rate_limiter: RedisRateLimiter | None = None,
) -> ChatOpenAI:
    """The primary model for a tier. Raises if no OpenRouter key is configured."""
    orc = settings.openrouter
    if orc.api_key is None:
        raise ValueError("OPENROUTER_API_KEY is not set")
    limiter = rate_limiter or _shared_rate_limiter(settings.infra.redis_url, orc.rpm, orc.rpd)
    return ChatOpenAI(
        model_name=settings.llm.model_for(tier),
        openai_api_base=orc.base_url,
        openai_api_key=orc.api_key,
        temperature=settings.llm.temperature,
        request_timeout=orc.timeout_s,
        max_retries=orc.max_retries,
        rate_limiter=limiter,
        default_headers={"X-Title": orc.app_title},
    )


def build_ollama_model(settings: Settings) -> ChatOpenAI:
    """The local fallback model, via Ollama's OpenAI-compatible endpoint (no rate limit)."""
    oll = settings.ollama
    return ChatOpenAI(
        model_name=oll.model,
        openai_api_base=oll.openai_base_url,
        openai_api_key=oll.api_key,
        temperature=settings.llm.temperature,
        request_timeout=oll.timeout_s,
        max_retries=0,
    )


def _shape(
    model: BaseChatModel,
    tools: Sequence[ToolLike] | None,
    schema: type[BaseModel] | None,
    method: StructuredMethod,
) -> Runnable[LanguageModelInput, Any]:
    if schema is not None:
        return model.with_structured_output(schema, method=method)
    if tools:
        return model.bind_tools(list(tools))
    return model


def get_llm(
    tier: LlmTier,
    *,
    tools: Sequence[ToolLike] | None = None,
    schema: type[BaseModel] | None = None,
    method: StructuredMethod = "function_calling",
    settings: Settings | None = None,
    primary: BaseChatModel | None = None,
    fallback: BaseChatModel | None = None,
) -> Runnable[LanguageModelInput, Any]:
    """Return the runnable for a tier.

    - `tools`: bind tools (the result is an AIMessage that may contain tool calls).
    - `schema`: return a validated instance of this Pydantic model instead of a message.
      `function_calling` is the default because every model we pin must support tools,
      while native JSON-schema support varies between providers.
    - `primary` / `fallback`: inject models (tests, or a different provider later).
    """
    if tools and schema is not None:
        raise ValueError("pass either tools or schema, not both")
    cfg = settings or get_settings()

    if primary is None and cfg.openrouter.api_key is not None:
        primary = build_openrouter_model(tier, cfg)

    if tier == "judge":
        if primary is None:
            raise ValueError("the judge tier needs OpenRouter: OPENROUTER_API_KEY is not set")
        return _shape(primary, tools, schema, method)

    local = fallback or build_ollama_model(cfg)
    if primary is None:
        logger.warning("OPENROUTER_API_KEY not set; tier %r runs on Ollama only", tier)
        return _shape(local, tools, schema, method)

    return _shape(primary, tools, schema, method).with_fallbacks(
        [_shape(local, tools, schema, method)]
    )


def served_by(message: BaseMessage) -> str:
    """Model that actually produced a message (OpenRouter returns the resolved model ID)."""
    meta = message.response_metadata or {}
    return str(meta.get("model_name") or meta.get("model") or "unknown")
