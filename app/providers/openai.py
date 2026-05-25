"""OpenAI provider."""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from . import providers
from .base import LLMProvider


@providers.register("openai")
class OpenAIProvider(LLMProvider):
    name = "OpenAI GPT-4o mini"
    env_key = "OPENAI_API_KEY"
    env_model = "OPENAI_MODEL"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        super().__init__(api_key, model)
        self._client = OpenAI(api_key=self.api_key)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
