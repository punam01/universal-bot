"""RAG orchestrator — connectors + query rewriters + indexers + rerankers + providers."""
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
from query_rewriters import query_rewriters
from query_rewriters.base import QueryRewriter
from rerankers import rerankers
from rerankers.base import Reranker


_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions from indexed documents. "
    "Answer strictly from the context provided with each user question. "
    "Cite sources inline as [1], [2], etc. If the answer is not in the "
    "context, say so honestly. Prior chat turns are for conversational "
    "continuity only — they do not contain authoritative facts."
)


def _merge_dedupe(batches: list[list[dict]], limit: int) -> list[dict]:
    """Merge multiple result lists; dedupe by (source, first 200 chars), keep max score."""
    seen: dict[tuple[str, str], dict] = {}
    for batch in batches:
        for r in batch:
            key = (r.get("source", "?"), r["text"][:200])
            existing = seen.get(key)
            if existing is None or r["score"] > existing["score"]:
                seen[key] = r
    return sorted(seen.values(), key=lambda x: -x["score"])[:limit]


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
        self._reranker_cache: dict[str, Reranker] = {}
        self._rewriter_cache: dict[str, QueryRewriter] = {}

    # ---------- registry surfaces (consumed by the UI) ----------

    def available_providers(self) -> list[tuple[str, str]]:
        return [(key, providers.get(key).name) for key in providers.available()]

    def available_connectors(self) -> list[tuple[str, str]]:
        return [(key, connectors.get(key).name) for key in connectors.keys()]

    def available_indexers(self) -> list[tuple[str, str]]:
        return [(key, indexers.get(key).name) for key in indexers.keys()]

    def available_rerankers(self) -> list[tuple[str, str]]:
        return [(key, rerankers.get(key).name) for key in rerankers.keys()]

    def available_rewriters(self) -> list[tuple[str, str]]:
        return [(key, query_rewriters.get(key).name) for key in query_rewriters.keys()]

    def connector_kind(self, key: str) -> str:
        return connectors.get(key).input_kind

    # ---------- memoized plugin getters ----------

    def _provider(self, key: str) -> LLMProvider:
        if key not in self._provider_cache:
            self._provider_cache[key] = providers.get(key).from_env()
        return self._provider_cache[key]

    def _indexer(self, key: str) -> Indexer:
        if key not in self._indexer_cache:
            cls = indexers.get(key)
            dep_keys = getattr(cls, "deps", []) or []
            if dep_keys:
                resolved = {dep: self._indexer(dep) for dep in dep_keys}
                self._indexer_cache[key] = cls(self._indexer_ctx, deps=resolved)
            else:
                self._indexer_cache[key] = cls(self._indexer_ctx)
        return self._indexer_cache[key]

    def _connector(self, key: str) -> Connector:
        if key not in self._connector_cache:
            self._connector_cache[key] = connectors.get(key)()
        return self._connector_cache[key]

    def _reranker(self, key: str) -> Reranker:
        if key not in self._reranker_cache:
            self._reranker_cache[key] = rerankers.get(key)()
        return self._reranker_cache[key]

    def _rewriter(self, key: str) -> QueryRewriter:
        if key not in self._rewriter_cache:
            self._rewriter_cache[key] = query_rewriters.get(key)()
        return self._rewriter_cache[key]

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
        self,
        indexer_key: str,
        query: str,
        top_k: int = 5,
        reranker_key: str = "none",
        rewriter_key: str = "none",
        provider_key: str | None = None,
    ) -> list[dict]:
        # 1. Decide what to actually query for.
        if rewriter_key == "none":
            queries = [query]
        else:
            if not provider_key:
                # Without an LLM we can't rewrite; fall back to the raw query.
                queries = [query]
            else:
                try:
                    queries = self._rewriter(rewriter_key).rewrite(
                        query, self._provider(provider_key)
                    )
                except Exception:
                    queries = [query]
                if not queries:
                    queries = [query]

        # 2. Retrieve from the indexer for each query. Wider pool if reranking.
        per_query_k = top_k * 4 if reranker_key != "none" else top_k
        indexer = self._indexer(indexer_key)
        batches = [indexer.retrieve(q, per_query_k) for q in queries]

        # 3. Merge if we issued multiple queries.
        if len(batches) == 1:
            candidates = batches[0]
        else:
            candidates = _merge_dedupe(batches, per_query_k)

        # 4. Rerank using the ORIGINAL query (not the rewrites).
        if reranker_key != "none" and candidates:
            candidates = self._reranker(reranker_key).rerank(query, candidates, top_k)
        elif len(candidates) > top_k:
            candidates = candidates[:top_k]

        return candidates

    def chat_stream(
        self,
        provider_key: str,
        query: str,
        sources: list[dict],
        history: list[dict] | None = None,
        max_history_msgs: int = 6,
    ) -> Iterator[str]:
        if not sources:
            yield (
                "I don't have any indexed documents yet. "
                "Add a source in the sidebar first."
            )
            return

        history = history or []
        recent = [
            h for h in history[-max_history_msgs:]
            if h.get("role") in ("user", "assistant") and h.get("content")
        ]

        context = "\n\n".join(
            f"[{i}] (from {s['source']}, page {s.get('page', '?')})\n{s['text']}"
            for i, s in enumerate(sources, 1)
        )

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for h in recent:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({
            "role": "user",
            "content": f"Context for this question:\n{context}\n\nQuestion: {query}",
        })

        yield from self._provider(provider_key).stream(messages)

    # ---------- source management ----------

    def list_all_sources(self) -> list[dict]:
        merged: dict[str, dict] = {}
        for key in indexers.keys():
            cls = indexers.get(key)
            if getattr(cls, "is_composite", False):
                continue
            idx = self._indexer(key)
            for entry in idx.list_sources():
                src = entry["source"]
                bucket = merged.setdefault(
                    src, {"source": src, "indexers": [], "chunks": {}, "total": 0}
                )
                if key not in bucket["indexers"]:
                    bucket["indexers"].append(key)
                bucket["chunks"][key] = entry["chunks"]
                bucket["total"] += entry["chunks"]
        return sorted(merged.values(), key=lambda x: x["source"].lower())

    def delete_source(self, source: str) -> int:
        total = 0
        for key in indexers.keys():
            cls = indexers.get(key)
            if getattr(cls, "is_composite", False):
                continue
            total += self._indexer(key).delete_source(source)
        return total

    def reset_all(self) -> None:
        for key in indexers.keys():
            self._indexer(key).reset()
