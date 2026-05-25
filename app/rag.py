"""RAG orchestrator with multi-project isolation.

A "project" is a named, isolated knowledge base — its own Chroma collections
and its own BM25 pickle file. The current project is held on the engine and
flows through `_indexer()` so every retrieval / ingest / source-management
call is scoped to that project.

The 'default' project deliberately keeps the legacy storage paths so
existing data from before this change is preserved.
"""
from __future__ import annotations

import json
import re
import shutil
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

_DEFAULT_PROJECT = "default"
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


def _merge_dedupe(batches: list[list[dict]], limit: int) -> list[dict]:
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
        self._current_project: str = _DEFAULT_PROJECT

        self._provider_cache: dict[str, LLMProvider] = {}
        # Indexer + connector caches keyed by (plugin_key, project) so switching
        # projects doesn't reuse the wrong storage.
        self._indexer_cache: dict[tuple[str, str], Indexer] = {}
        self._connector_cache: dict[str, Connector] = {}
        self._reranker_cache: dict[str, Reranker] = {}
        self._rewriter_cache: dict[str, QueryRewriter] = {}

        # Ensure the projects index file lists 'default' at minimum.
        self._ensure_project_in_index(_DEFAULT_PROJECT)

    # ---------- project management ----------

    @property
    def current_project(self) -> str:
        return self._current_project

    def _projects_index_path(self) -> Path:
        return Path(settings.chroma_dir) / "projects.json"

    def list_projects(self) -> list[str]:
        path = self._projects_index_path()
        if not path.exists():
            return [_DEFAULT_PROJECT]
        try:
            with path.open() as f:
                data = json.load(f)
            projects = sorted({_DEFAULT_PROJECT, *data})
            return projects
        except Exception:
            return [_DEFAULT_PROJECT]

    def _ensure_project_in_index(self, name: str) -> None:
        existing = set(self.list_projects())
        if name in existing:
            return
        existing.add(name)
        path = self._projects_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(sorted(existing), f)

    def _validate_project_name(self, name: str) -> str:
        name = name.strip()
        if not _PROJECT_NAME_RE.match(name):
            raise ValueError(
                "Project name must start with a letter or digit and contain "
                "only letters, digits, '.', '_', or '-' (max 32 chars)."
            )
        return name

    def create_project(self, name: str) -> str:
        name = self._validate_project_name(name)
        self._ensure_project_in_index(name)
        return name

    def set_project(self, name: str) -> None:
        if name not in self.list_projects():
            raise ValueError(f"Unknown project: {name!r}")
        self._current_project = name

    def delete_project(self, name: str) -> None:
        if name == _DEFAULT_PROJECT:
            raise ValueError("The 'default' project cannot be deleted.")
        if name not in self.list_projects():
            raise ValueError(f"Unknown project: {name!r}")

        # Drop chroma collections owned by this project.
        try:
            for col in self._chroma.list_collections():
                if col.name.startswith(f"{settings.collection}__{name}__"):
                    try:
                        self._chroma.delete_collection(col.name)
                    except Exception:
                        pass
        except Exception:
            pass

        # Wipe its on-disk storage (bm25 pickle, etc.).
        project_dir = Path(settings.chroma_dir) / name
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)

        # Update the projects index.
        path = self._projects_index_path()
        if path.exists():
            try:
                with path.open() as f:
                    data = json.load(f)
                remaining = sorted({p for p in data if p != name})
                with path.open("w") as f:
                    json.dump(remaining, f)
            except Exception:
                pass

        # Drop cached indexer instances for this project.
        self._indexer_cache = {
            key: val
            for key, val in self._indexer_cache.items()
            if key[1] != name
        }
        if self._current_project == name:
            self._current_project = _DEFAULT_PROJECT

    # ---------- per-project IndexerContext ----------

    def _make_ctx(self, project: str) -> IndexerContext:
        if project == _DEFAULT_PROJECT:
            # Preserve the original storage layout so data from before
            # project isolation keeps working.
            return IndexerContext(
                chroma_client=self._chroma,
                embedder=self._embedder,
                collection_base=settings.collection,
                storage_dir=Path(settings.chroma_dir),
            )
        return IndexerContext(
            chroma_client=self._chroma,
            embedder=self._embedder,
            collection_base=f"{settings.collection}__{project}",
            storage_dir=Path(settings.chroma_dir) / project,
        )

    # ---------- registry surfaces (consumed by the UI) ----------

    def available_providers(self) -> list[tuple[str, str]]:
        return [(key, providers.get(key).name) for key in providers.available()]

    def available_connectors(self) -> list[tuple[str, str]]:
        return [(key, connectors.get(key).name) for key in connectors.available()]

    def connector_extensions(self, key: str) -> list[str]:
        """File extensions the named connector accepts (for the file uploader)."""
        return list(connectors.get(key).accepted_extensions or [])

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
        project = self._current_project
        cache_key = (key, project)
        if cache_key not in self._indexer_cache:
            cls = indexers.get(key)
            ctx = self._make_ctx(project)
            dep_keys = getattr(cls, "deps", []) or []
            if dep_keys:
                resolved = {dep: self._indexer(dep) for dep in dep_keys}
                self._indexer_cache[cache_key] = cls(ctx, deps=resolved)
            else:
                self._indexer_cache[cache_key] = cls(ctx)
        return self._indexer_cache[cache_key]

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

    def get_provider(self, key: str) -> LLMProvider:
        return self._provider(key)

    # ---------- pipeline ----------

    def ingest(
        self,
        connector_key: str,
        indexer_key: str,
        payload,
        source_name: str,
        on_progress=None,
    ) -> int:
        """Ingest a payload via the chosen connector + indexer.

        `on_progress`, if provided, is called as `on_progress(count, source)`
        for every Document yielded by the connector. Useful for crawls that
        emit one Document per page — lets the UI show "fetched N pages so far".
        """
        raw_docs = self._connector(connector_key).fetch(payload, source_name)

        if on_progress is None:
            docs = raw_docs
        else:
            def _with_progress():
                count = 0
                for doc in raw_docs:
                    count += 1
                    try:
                        on_progress(count, doc.source)
                    except Exception:
                        # never break ingest because the UI hook misbehaved
                        pass
                    yield doc

            docs = _with_progress()

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
        if rewriter_key == "none":
            queries = [query]
        else:
            if not provider_key:
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

        per_query_k = top_k * 4 if reranker_key != "none" else top_k
        indexer = self._indexer(indexer_key)
        batches = [indexer.retrieve(q, per_query_k) for q in queries]

        if len(batches) == 1:
            candidates = batches[0]
        else:
            candidates = _merge_dedupe(batches, per_query_k)

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

    # ---------- source management (scoped to current project) ----------

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

    # ---------- corpus advisor ----------

    def sample_chunks(self, n: int = 8) -> list[str]:
        try:
            semantic = self._indexer("semantic")
            results = semantic._collection.get(limit=n, include=["documents"])
            return results.get("documents") or []
        except Exception:
            return []

    def recommend_settings(self, user_goals: str, provider_key: str) -> dict:
        from advisor import recommend
        return recommend(self, user_goals, provider_key)
