"""Ollama provider — local LLM runtime, no API key needed."""
from __future__ import annotations

import json
import os
from typing import Iterator

import httpx

from . import providers
from .base import LLMProvider


@providers.register("ollama")
class OllamaProvider(LLMProvider):
    name = "Ollama (local)"
    env_model = "OLLAMA_MODEL"
    default_model = "qwen2.5:7b-instruct"
    requires_key = False  # local, no key

    def __init__(self, api_key: str = "", model: str | None = None) -> None:
        super().__init__(api_key, model)
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("done"):
                        return
                    content = obj.get("message", {}).get("content", "")
                    if content:
                        yield content
