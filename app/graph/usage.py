"""Record which model actually answered each LLM call (primary or Ollama fallback)."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult


def undouble(name: str) -> str:
    """Streaming merges chunk metadata by string concatenation ("a/ba/b" -> "a/b")."""
    while len(name) % 2 == 0 and name and name[: len(name) // 2] == name[len(name) // 2 :]:
        name = name[: len(name) // 2]
    return name


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
                        clean = undouble(str(name))
                        if clean not in self.models:
                            self.models.append(clean)
