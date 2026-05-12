"""Hybrid indexer — composes semantic + syntactic via Reciprocal Rank Fusion."""
from __future__ import annotations

from typing import Iterable

from connectors.base import Document

from . import indexers
from .base import Indexer, IndexerContext
from .semantic import SemanticIndexer
from .syntactic import SyntacticIndexer


def _rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion. Each input list is presumed sorted best-first.

    The fused score is sum_over_lists( 1 / (k + rank_in_list) ).
    Identity is (source, first 200 chars of text) so chunks shared across
    indexers collapse.
    """
    scores: dict[str, float] = {}
    canonical: dict[str, dict] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = f"{result.get('source', '?')}|{result['text'][:200]}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            canonical.setdefault(key, result)
    sorted_keys = sorted(scores, key=lambda x: -scores[x])
    return [
        {**canonical[key], "score": scores[key]}
        for key in sorted_keys
    ]


@indexers.register("hybrid")
class HybridIndexer(Indexer):
    name = "Hybrid (semantic + BM25, RRF)"
    description = (
        "Indexes into BOTH semantic and syntactic; at query time, fuses "
        "results via Reciprocal Rank Fusion. Best for mixed corpora."
    )

    def __init__(self, ctx: IndexerContext) -> None:
        super().__init__(ctx)
        self._semantic = SemanticIndexer(ctx)
        self._syntactic = SyntacticIndexer(ctx)

    def index(self, docs: Iterable[Document], chunk_size: int, overlap: int) -> int:
        doc_list = list(docs)
        n = self._semantic.index(iter(doc_list), chunk_size, overlap)
        self._syntactic.index(iter(doc_list), chunk_size, overlap)
        return n  # same N from both — chunking is identical

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        wider = max(top_k * 4, 20)
        sem = self._semantic.retrieve(query, wider)
        syn = self._syntactic.retrieve(query, wider)
        return _rrf([sem, syn])[:top_k]

    def reset(self) -> None:
        self._semantic.reset()
        self._syntactic.reset()
