"""Phase 1 runtime config — reads env vars from .env in the project root.

Supports two LLM providers, both free-tier-friendly:
- groq    : Groq (fast Llama 3.3 70B free tier — recommended)
- gemini  : Google AI Studio Gemini 2.0 Flash
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Auto-load .env from the project root (one level up from this file's dir).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)


@dataclass(frozen=True)
class Settings:
    chroma_dir: str
    collection: str
    llm_provider: str
    llm_api_key: str
    llm_model: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int


_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}


def load() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in _DEFAULT_MODELS:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER='{provider}'. "
            f"Use one of: {', '.join(_DEFAULT_MODELS)}."
        )

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        key_source = "GOOGLE_API_KEY (https://aistudio.google.com/app/apikey)"
    else:  # groq
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        key_source = "GROQ_API_KEY (https://console.groq.com/keys)"

    if not api_key:
        raise RuntimeError(
            f"LLM_PROVIDER={provider} requires {key_source}. Set it in .env."
        )

    return Settings(
        chroma_dir=os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "app" / "chroma_data")),
        collection=os.getenv("CHROMA_COLLECTION", "documents"),
        llm_provider=provider,
        llm_api_key=api_key,
        llm_model=os.getenv("LLM_MODEL", _DEFAULT_MODELS[provider]),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )


settings = load()
