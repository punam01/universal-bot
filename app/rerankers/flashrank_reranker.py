"""FlashRank reranker — tiny CPU model, big quality lift."""
from __future__ import annotations

from flashrank import Ranker, RerankRequest

from . import rerankers
from .base import Reranker


@rerankers.register("flashrank")
class FlashRankReranker(Reranker):
    name = "FlashRank (ms-marco MiniLM-L-12)"
    description = "Tiny CPU reranker (~4 MB). Big quality boost for almost no cost."

    def __init__(self) -> None:
        # Lazy: model downloads on first rerank() call.
        self._ranker: Ranker | None = None

    def _get(self) -> Ranker:
        if self._ranker is None:
            self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not candidates:
            return []
        passages = [
            {"id": i, "text": c["text"], "meta": c}
            for i, c in enumerate(candidates)
        ]
        results = self._get().rerank(RerankRequest(query=query, passages=passages))
        return [
            {**r["meta"], "score": float(r["score"])}
            for r in results[:top_k]
        ]
