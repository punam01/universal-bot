"""Semantic indexer — dense vectors in ChromaDB."""
from __future__ import annotations

import uuid
from typing import Iterable

from connectors.base import Document

from . import indexers
from ._chunking import sliding_window
from .base import Indexer, IndexerContext


@indexers.register("semantic")
class SemanticIndexer(Indexer):
    name = "Semantic (vector search)"
    description = "Best for paraphrased questions and concept-level retrieval."

    def __init__(self, ctx: IndexerContext) -> None:
        super().__init__(ctx)
        self._collection_name = f"{ctx.collection_base}_semantic"
        self._collection = self._get_or_create()

    def _get_or_create(self):
        return self.ctx.chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, docs: Iterable[Document], chunk_size: int, overlap: int) -> int:
        pairs: list[tuple[Document, str]] = []
        for doc in docs:
            for chunk in sliding_window(doc.text, chunk_size, overlap):
                pairs.append((doc, chunk))

        if not pairs:
            return 0

        vectors = list(self.ctx.embedder.embed([chunk for _, chunk in pairs]))
        self._collection.add(
            ids=[str(uuid.uuid4()) for _ in pairs],
            embeddings=[v.tolist() for v in vectors],
            documents=[chunk for _, chunk in pairs],
            metadatas=[
                {"source": d.source, "page": d.page if d.page is not None else -1}
                for d, _ in pairs
            ],
        )
        return len(pairs)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        v = next(self.ctx.embedder.embed([query]))
        results = self._collection.query(
            query_embeddings=[v.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results.get("ids") or not results["ids"][0]:
            return []
        return [
            {
                "text": doc,
                "source": meta.get("source", "?"),
                "page": meta.get("page") if (meta.get("page") or -1) > 0 else None,
                "score": float(1.0 - dist),  # cosine distance → similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def reset(self) -> None:
        self.ctx.chroma_client.delete_collection(name=self._collection_name)
        self._collection = self._get_or_create()
