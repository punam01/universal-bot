"""LLM provider plugin contract.

Each provider:
- declares its env vars (api key and optional model override)
- implements stream(messages) -> Iterator[str]
- gets is_available() and from_env() for free

`messages` is a list of {role, content} dicts, where role is one of
'system', 'user', or 'assistant'. Providers that don't natively support
a 'system' role should fold its content into the first user message or
into a provider-specific system_instruction field.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import ClassVar, Iterator


class LLMProvider(ABC):
    name: ClassVar[str]                   # display name (UI)
    env_key: ClassVar[str]                # env var holding the API key
    default_model: ClassVar[str]          # fallback model name
    env_model: ClassVar[str] = ""         # optional env var to override model

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or self.default_model

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.getenv(cls.env_key, "").strip())

    @classmethod
    def from_env(cls) -> "LLMProvider":
        api_key = os.getenv(cls.env_key, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{cls.env_key} is not set; cannot use provider '{cls.__name__}'."
            )
        model = None
        if cls.env_model:
            model = os.getenv(cls.env_model, "").strip() or None
        return cls(api_key=api_key, model=model)

    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield completion tokens given a chat-style messages list."""
