"""Phase 1 RAG engine (native, no Docker).

Pipeline:
    PDF bytes → text per page → sliding-window chunks → fastembed vectors → ChromaDB
    Query     → embed → ChromaDB search → context assembly → LLM stream → tokens

LLM provider is switchable via LLM_PROVIDER env var (groq | gemini).
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Iterator

import chromadb
import pypdfium2 as pdfium
from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding

from config import settings


class RAGEngine:
    def __init__(self) -> None:
        Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = TextEmbedding(model_name=settings.embed_model)

        # Conditional LLM client init — only the chosen provider's SDK is touched.
        if settings.llm_provider == "gemini":
            from google import genai
            self._gemini = genai.Client(api_key=settings.llm_api_key)
        elif settings.llm_provider == "groq":
            from groq import Groq
            self._groq = Groq(api_key=settings.llm_api_key)

    def reset(self) -> None:
        self.chroma.delete_collection(name=settings.collection)
        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ----- ingestion -----

    @staticmethod
    def _extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
        pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
        out: list[tuple[int, str]] = []
        for i in range(len(pdf)):
            text = pdf[i].get_textpage().get_text_range() or ""
            out.append((i + 1, text))
        return out

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= size:
            return [text]
        out: list[str] = []
        start, n = 0, len(text)
        while start < n:
            end = min(start + size, n)
            piece = text[start:end].strip()
            if piece:
                out.append(piece)
            if end == n:
                break
            start = end - overlap
        return out

    def ingest_pdf(self, pdf_bytes: bytes, source_name: str) -> int:
        records: list[dict] = []
        for page_num, page_text in self._extract_pages(pdf_bytes):
            for chunk in self._chunk(page_text, settings.chunk_size, settings.chunk_overlap):
                records.append({"text": chunk, "page": page_num, "source": source_name})

        if not records:
            return 0

        vectors = list(self.embedder.embed([r["text"] for r in records]))
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in records],
            embeddings=[v.tolist() for v in vectors],
            documents=[r["text"] for r in records],
            metadatas=[{"source": r["source"], "page": r["page"]} for r in records],
        )
        return len(records)

    # ----- retrieval + generation -----

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        v = next(self.embedder.embed([query]))
        results = self.collection.query(
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
                "page": meta.get("page"),
                "score": float(1.0 - dist),  # cosine distance → similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def chat_stream(self, query: str, sources: list[dict]) -> Iterator[str]:
        if not sources:
            yield (
                "I don't have any indexed documents yet. "
                "Upload a PDF in the sidebar first."
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

        if settings.llm_provider == "gemini":
            yield from self._stream_gemini(prompt)
        else:
            yield from self._stream_groq(prompt)

    def _stream_gemini(self, prompt: str) -> Iterator[str]:
        stream = self._gemini.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _stream_groq(self, prompt: str) -> Iterator[str]:
        stream = self._groq.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
