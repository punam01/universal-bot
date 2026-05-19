"""Query rewriter plugin contract.

A rewriter transforms a user question into one or more queries to actually
retrieve with. It uses an LLM (the same provider chosen for chat) and runs
BEFORE retrieval. Optional — selecting 'none' skips it.

If a rewriter returns multiple queries, the engine retrieves for each and
merges the results (dedup by source+text, keep max score).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from providers.base import LLMProvider


class QueryRewriter(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abstractmethod
    def rewrite(self, query: str, llm: "LLMProvider") -> list[str]:
        """Return the list of queries to retrieve with."""
