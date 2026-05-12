"""Reranker plugin contract — reorders candidate chunks by relevance to a query."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class Reranker(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Return the top_k most relevant candidates, each with a 'score' field."""
