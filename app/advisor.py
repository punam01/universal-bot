"""Corpus advisor — analyzes the indexed content and recommends pipeline settings.

Single-shot for now (one LLM call). Could be expanded to a multi-turn LangGraph
agent in a later phase if recommendation quality demands clarifying questions.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag import RAGEngine


_INDEXER_DESCRIPTIONS = (
    "- semantic    : vector similarity. Best for paraphrased, conceptual questions.\n"
    "- syntactic   : BM25 keyword. Best for exact codes, names, identifiers.\n"
    "- hybrid      : both fused via RRF. Best for mixed corpora.\n"
)

_REWRITER_DESCRIPTIONS = (
    "- none        : skip rewriting. Cheapest, lowest latency.\n"
    "- hyde        : generate a hypothetical answer and retrieve using that.\n"
    "                Best for vague or under-specified questions.\n"
    "- multi_query : paraphrase the question 3 ways and merge results.\n"
    "                Best when the corpus uses different terminology than users.\n"
)

_RERANKER_DESCRIPTIONS = (
    "- none        : raw retrieval results.\n"
    "- flashrank   : small CPU reranker. Big quality lift, ~50ms overhead.\n"
    "                Recommended unless latency-critical.\n"
)

_PROMPT_TEMPLATE = """You are an indexing strategist for a RAG system. Given a sample
of an already-indexed corpus and the user's stated goals, pick the best pipeline
settings from the options below.

Indexers:
{indexers}

Query rewriters:
{rewriters}

Rerankers:
{rerankers}

CORPUS PROFILE
- Total chunks indexed: {total_chunks}
- Number of sources:    {num_sources}
- Source names sample:  {source_names}

SAMPLE CHUNKS (first 500 chars of each)
{samples}

USER GOALS
{goals}

Reply with exactly one JSON object on a single line — no code fences,
no markdown, no commentary. The object must have these fields:

{{"indexer": "semantic"|"syntactic"|"hybrid",
  "rewriter": "none"|"hyde"|"multi_query",
  "reranker": "none"|"flashrank",
  "chunk_size": <int between 200 and 3000>,
  "chunk_overlap": <int between 0 and 500>,
  "rationale": "<2 to 4 sentences explaining the picks for THIS corpus>"}}
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    return text.strip()


def _find_json_object(text: str) -> str | None:
    """Best-effort: find the first balanced { ... } block in text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def recommend(engine: "RAGEngine", user_goals: str, provider_key: str) -> dict:
    """Return a recommendation dict.

    On success:  {"recommendation": {indexer, rewriter, reranker, chunk_size,
                                     chunk_overlap, rationale}, "raw": "..."}
    On no data:  {"error": "..."}
    On bad parse: {"error": "...", "raw": "..."}
    """
    sources = engine.list_all_sources()
    if not sources:
        return {"error": "No documents indexed yet. Add at least one source first."}

    total_chunks = sum(s["total"] for s in sources)
    source_names = [s["source"] for s in sources[:5]]
    if len(sources) > 5:
        source_names.append(f"... and {len(sources) - 5} more")

    samples = engine.sample_chunks(n=8)
    if not samples:
        return {
            "error": (
                "Found sources, but no semantic chunks to sample. Re-index with "
                "the 'semantic' or 'hybrid' indexer first."
            )
        }

    sample_text = "\n---\n".join(
        f"[{i+1}] {s[:500]}" for i, s in enumerate(samples)
    )

    prompt = _PROMPT_TEMPLATE.format(
        indexers=_INDEXER_DESCRIPTIONS,
        rewriters=_REWRITER_DESCRIPTIONS,
        rerankers=_RERANKER_DESCRIPTIONS,
        total_chunks=total_chunks,
        num_sources=len(sources),
        source_names=", ".join(source_names),
        samples=sample_text,
        goals=user_goals.strip() or "(not specified — assume general Q&A over the corpus)",
    )

    llm = engine.get_provider(provider_key)
    raw = "".join(llm.stream([{"role": "user", "content": prompt}]))

    candidate = _strip_code_fences(raw)
    parsed = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        block = _find_json_object(candidate)
        if block:
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return {"error": "LLM did not return valid JSON.", "raw": raw}

    required = {"indexer", "rewriter", "reranker", "chunk_size", "chunk_overlap", "rationale"}
    missing = required - parsed.keys()
    if missing:
        return {
            "error": f"Recommendation is missing fields: {sorted(missing)}",
            "raw": raw,
        }

    return {"recommendation": parsed, "raw": raw}
