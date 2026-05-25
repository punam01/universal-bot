"""Anthropic Claude provider."""
from __future__ import annotations

from typing import Iterator

from anthropic import Anthropic

from . import providers
from .base import LLMProvider


@providers.register("anthropic")
class AnthropicProvider(LLMProvider):
    name = "Anthropic Claude Sonnet"
    env_key = "ANTHROPIC_API_KEY"
    env_model = "ANTHROPIC_MODEL"
    default_model = "claude-sonnet-4-5"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        super().__init__(api_key, model)
        self._client = Anthropic(api_key=self.api_key)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        # Anthropic separates the system prompt from the chat transcript.
        system_text = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        chat_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") != "system"
        ]
        with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system_text or None,
            messages=chat_msgs,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
