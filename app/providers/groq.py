"""Groq provider — fast Llama 3.3 70B free tier."""
from __future__ import annotations

from typing import Iterator

from groq import Groq

from . import providers
from .base import LLMProvider


@providers.register("groq")
class GroqProvider(LLMProvider):
    name = "Groq (Llama 3.3 70B)"
    env_key = "GROQ_API_KEY"
    env_model = "GROQ_MODEL"
    default_model = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        super().__init__(api_key, model)
        self._client = Groq(api_key=self.api_key)

    def stream(self, prompt: str) -> Iterator[str]:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content
