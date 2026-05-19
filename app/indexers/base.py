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
    is_composite: ClassVar[bool] = False     # True if it wraps other indexers
    deps: ClassVar[list[str]] = []            # keys of other indexers it depends on

    def __init__(
        self,
        ctx: IndexerContext,
        deps: dict[str, "Indexer"] | None = None,
    ) -> None:
        self.ctx = ctx
        self._deps = deps or {}

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

    @abstractmethod
    def list_sources(self) -> list[dict]:
        """Return [{source: str, chunks: int}] for everything currently indexed."""

    @abstractmethod
    def delete_source(self, source: str) -> int:
        """Delete every chunk whose source matches. Returns count removed."""
