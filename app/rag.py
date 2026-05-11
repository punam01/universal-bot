"""Phase 2 RAG orchestrator — wires connector + indexer + provider plugins.

This file no longer knows anything about Groq, Gemini, PDFs, ChromaDB, or
BM25 — those live inside their respective plugin packages. It only knows
the registries and the high-level pipeline (ingest / retrieve / chat).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding

from config import settings
from connectors import connectors, Connector
from indexers import indexers, Indexer, IndexerContext
from providers import providers
from providers.base import LLMProvider


class RAGEngine:
    def __init__(self) -> None:
        Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = TextEmbedding(model_name=settings.embed_model)
        self._indexer_ctx = IndexerContext(
            chroma_client=self._chroma,
            embedder=self._embedder,
            collection_base=settings.collection,
            storage_dir=Path(settings.chroma_dir),
        )
        self._provider_cache: dict[str, LLMProvider] = {}
        self._indexer_cache: dict[str, Indexer] = {}
        self._connector_cache: dict[str, Connector] = {}

    # ---------- registry surfaces (consumed by the UI) ----------

    def available_providers(self) -> list[tuple[str, str]]:
        return [(key, providers.get(key).name) for key in providers.available()]

    def available_connectors(self) -> list[tuple[str, str]]:
        return [(key, connectors.get(key).name) for key in connectors.keys()]

    def available_indexers(self) -> list[tuple[str, str]]:
        return [(key, indexers.get(key).name) for key in indexers.keys()]

    # ---------- memoized plugin getters ----------

    def _provider(self, key: str) -> LLMProvider:
        if key not in self._provider_cache:
            self._provider_cache[key] = providers.get(key).from_env()
        return self._provider_cache[key]

    def _indexer(self, key: str) -> Indexer:
        if key not in self._indexer_cache:
            self._indexer_cache[key] = indexers.get(key)(self._indexer_ctx)
        return self._indexer_cache[key]

    def _connector(self, key: str) -> Connector:
        if key not in self._connector_cache:
            self._connector_cache[key] = connectors.get(key)()
        return self._connector_cache[key]

    # ---------- pipeline ----------

    def ingest(
        self,
        connector_key: str,
        indexer_key: str,
        payload,
        source_name: str,
    ) -> int:
        docs = self._connector(connector_key).fetch(payload, source_name)
        return self._indexer(indexer_key).index(
            docs, settings.chunk_size, settings.chunk_overlap
        )

    def retrieve(
        self, indexer_key: str, query: str, top_k: int = 5
    ) -> list[dict]:
        return self._indexer(indexer_key).retrieve(query, top_k)

    def chat_stream(
        self,
        provider_key: str,
        query: str,
        sources: list[dict],
    ) -> Iterator[str]:
        if not sources:
            yield (
                "I don't have any indexed documents yet. "
                "Add a source in the sidebar first."
            )
            return

        context = "\n\n".join(
            f"[{i}] (from {s['source']}, page {s.get('page', '?')})\n{s['text']}"
            for i, s in enumerate(sources, 1)
        )
        prompt = (
            "Answer the question strictly from the context below. "
            "Cite sources inline as [1], [2], etc. "
            "If the answer is not in the context, say so honestly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        yield from self._provider(provider_key).stream(prompt)

    def reset_all(self) -> None:
        for key in indexers.keys():
            self._indexer(key).reset()
