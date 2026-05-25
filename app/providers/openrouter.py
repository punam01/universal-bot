"""OpenRouter provider — OpenAI-compatible endpoint with many free models."""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from . import providers
from .base import LLMProvider


@providers.register("openrouter")
class OpenRouterProvider(LLMProvider):
    name = "OpenRouter"
    env_key = "OPENROUTER_API_KEY"
    env_model = "OPENROUTER_MODEL"
    # Default to a free-tier model; user can override via OPENROUTER_MODEL.
    default_model = "meta-llama/llama-3.3-70b-instruct:free"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        super().__init__(api_key, model)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def stream(self, messages: list[dict]) -> Iterator[str]:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
