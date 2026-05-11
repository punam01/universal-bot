"""Gemini provider — Google AI Studio free tier."""
from __future__ import annotations

from typing import Iterator

from google import genai

from . import providers
from .base import LLMProvider


@providers.register("gemini")
class GeminiProvider(LLMProvider):
    name = "Gemini 2.0 Flash"
    env_key = "GOOGLE_API_KEY"
    env_model = "GEMINI_MODEL"
    default_model = "gemini-2.0-flash"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        super().__init__(api_key, model)
        self._client = genai.Client(api_key=self.api_key)

    def stream(self, prompt: str) -> Iterator[str]:
        stream = self._client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
