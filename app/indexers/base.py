"""Indexer plugin contract — stores chunks and retrieves them by query."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb
    from fastembed import TextEmbedding
    from connectors.base import Document


@dataclass
class IndexerContext:
    """Shared resources injected into every indexer at construction time."""
    chroma_client: "chromadb.PersistentClient"
    embedder: "TextEmbedding"
    collection_base: str
    storage_dir: Path


class Indexer(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""

    def __init__(self, ctx: IndexerContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def index(
        self,
        docs: "Iterable[Document]",
        chunk_size: int,
        overlap: int,
    ) -> int:
        """Index documents. Return total chunks stored."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return [{text, source, page, score}, ...] ordered by relevance."""

    @abstractmethod
    def reset(self) -> None:
        """Remove all stored data for this indexer."""
