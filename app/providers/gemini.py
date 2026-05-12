"""Gemini provider — Google AI Studio free tier."""
from __future__ import annotations

from typing import Iterator

from google import genai
from google.genai import types

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

    def stream(self, messages: list[dict]) -> Iterator[str]:
        # Gemini has no 'system' role; concat system messages into the
        # provider's system_instruction config. Assistant -> 'model'.
        system_text = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        contents = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            mapped_role = "model" if role == "assistant" else "user"
            contents.append({"role": mapped_role, "parts": [{"text": m["content"]}]})

        config = (
            types.GenerateContentConfig(system_instruction=system_text)
            if system_text
            else None
        )

        stream = self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
