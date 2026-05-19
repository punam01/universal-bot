"""Syntactic indexer — BM25 keyword search (no embeddings)."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

from connectors.base import Document

from . import indexers
from ._chunking import sliding_window
from .base import Indexer, IndexerContext


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-'][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@indexers.register("syntactic")
class SyntacticIndexer(Indexer):
    name = "Syntactic (BM25 keyword)"
    description = "Best for exact identifiers, codes, names, terminology."

    def __init__(self, ctx: IndexerContext, deps=None) -> None:
        super().__init__(ctx, deps)
        self._path: Path = Path(ctx.storage_dir) / "bm25.pkl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._corpus: list[str] = []
        self._metas: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("rb") as f:
            data = pickle.load(f)
        self._corpus = data.get("corpus", [])
        self._metas = data.get("metas", [])
        if self._corpus:
            self._bm25 = BM25Okapi([_tokenize(c) for c in self._corpus])

    def _save(self) -> None:
        with self._path.open("wb") as f:
            pickle.dump({"corpus": self._corpus, "metas": self._metas}, f)

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25Okapi([_tokenize(c) for c in self._corpus]) if self._corpus else None

    def index(self, docs: Iterable[Document], chunk_size: int, overlap: int) -> int:
        added = 0
        for doc in docs:
            for chunk in sliding_window(doc.text, chunk_size, overlap):
                self._corpus.append(chunk)
                self._metas.append({"source": doc.source, "page": doc.page})
                added += 1
        if added:
            self._rebuild_bm25()
            self._save()
        return added

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._bm25 or not self._corpus:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: -pair[1])[:top_k]
        return [
            {
                "text": self._corpus[i],
                "source": self._metas[i]["source"],
                "page": self._metas[i].get("page"),
                "score": float(score),
            }
            for i, score in ranked
            if score > 0
        ]

    def reset(self) -> None:
        self._corpus = []
        self._metas = []
        self._bm25 = None
        if self._path.exists():
            self._path.unlink()

    def list_sources(self) -> list[dict]:
        counts: dict[str, int] = {}
        for meta in self._metas:
            src = meta.get("source", "?")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]

    def delete_source(self, source: str) -> int:
        keep_chunks: list[str] = []
        keep_metas: list[dict] = []
        removed = 0
        for chunk, meta in zip(self._corpus, self._metas):
            if meta.get("source") == source:
                removed += 1
            else:
                keep_chunks.append(chunk)
                keep_metas.append(meta)
        if removed:
            self._corpus = keep_chunks
            self._metas = keep_metas
            self._rebuild_bm25()
            self._save()
        return removed
