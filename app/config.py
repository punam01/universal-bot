"""Runtime config — global defaults only.

Provider API keys live with each provider plugin. The UI picks plugins
dynamically based on availability.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)


@dataclass(frozen=True)
class Settings:
    chroma_dir: str
    collection: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    default_provider: str
    default_indexer: str
    default_connector: str
    default_reranker: str
    default_rewriter: str


def load() -> Settings:
    return Settings(
        chroma_dir=os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "app" / "chroma_data")),
        collection=os.getenv("CHROMA_COLLECTION", "documents"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        default_provider=os.getenv("LLM_PROVIDER", "groq"),
        default_indexer=os.getenv("DEFAULT_INDEXER", "semantic"),
        default_connector=os.getenv("DEFAULT_CONNECTOR", "pdf"),
        default_reranker=os.getenv("DEFAULT_RERANKER", "none"),
        default_rewriter=os.getenv("DEFAULT_REWRITER", "none"),
    )


settings = load()
