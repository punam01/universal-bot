"""Hybrid indexer — composes semantic + syntactic via Reciprocal Rank Fusion.

This is a composite indexer: it does not own storage. It delegates to the
shared semantic and syntactic indexer instances provided by the engine,
so a single source uploaded via 'hybrid' is visible to standalone
'semantic' and 'syntactic' too (and source listing/deletion stay consistent).
"""
from __future__ import annotations

from typing import ClassVar, Iterable

from connectors.base import Document

from . import indexers
from .base import Indexer, IndexerContext
from .semantic import SemanticIndexer
from .syntactic import SyntacticIndexer


def _rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across result lists."""
    scores: dict[str, float] = {}
    canonical: dict[str, dict] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = f"{result.get('source', '?')}|{result['text'][:200]}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            canonical.setdefault(key, result)
    sorted_keys = sorted(scores, key=lambda x: -scores[x])
    return [{**canonical[key], "score": scores[key]} for key in sorted_keys]


@indexers.register("hybrid")
class HybridIndexer(Indexer):
    name = "Hybrid (semantic + BM25, RRF)"
    description = (
        "Indexes into BOTH semantic and syntactic; at query time fuses "
        "results via Reciprocal Rank Fusion. Best for mixed corpora."
    )
    is_composite: ClassVar[bool] = True
    deps: ClassVar[list[str]] = ["semantic", "syntactic"]

    def __init__(self, ctx: IndexerContext, deps=None) -> None:
        super().__init__(ctx, deps)
        # Reuse injected instances when present (the engine wires this up).
        # Fall back to local construction so the class still works standalone.
        self._semantic: SemanticIndexer = (
            self._deps.get("semantic") if self._deps else None
        ) or SemanticIndexer(ctx)
        self._syntactic: SyntacticIndexer = (
            self._deps.get("syntactic") if self._deps else None
        ) or SyntacticIndexer(ctx)

    def index(self, docs: Iterable[Document], chunk_size: int, overlap: int) -> int:
        doc_list = list(docs)
        n = self._semantic.index(iter(doc_list), chunk_size, overlap)
        self._syntactic.index(iter(doc_list), chunk_size, overlap)
        return n

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        wider = max(top_k * 4, 20)
        sem = self._semantic.retrieve(query, wider)
        syn = self._syntactic.retrieve(query, wider)
        return _rrf([sem, syn])[:top_k]

    def reset(self) -> None:
        self._semantic.reset()
        self._syntactic.reset()

    def list_sources(self) -> list[dict]:
        # Both sub-indexers see the same content under hybrid; defer to semantic.
        return self._semantic.list_sources()

    def delete_source(self, source: str) -> int:
        n = self._semantic.delete_source(source)
        self._syntactic.delete_source(source)
        return n
