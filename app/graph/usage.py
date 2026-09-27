"""Record which model actually answered each LLM call (primary or Ollama fallback)."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult


class ModelUsageRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.models: list[str] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                if isinstance(gen, ChatGeneration):
                    meta = gen.message.response_metadata or {}
                    name = meta.get("model_name") or meta.get("model")
                    if name:
                        self.models.append(str(name))
