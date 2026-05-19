"""HyDE — Hypothetical Document Embeddings.

Generates a confident-sounding hypothetical answer to the question with the
LLM, then retrieves using that fake answer as the query. The hypothetical
answer typically lives closer to relevant passages in vector space than
the question itself, so retrieval improves for vague or under-specified
queries.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (2022).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import query_rewriters
from .base import QueryRewriter

if TYPE_CHECKING:
    from providers.base import LLMProvider


_HYDE_PROMPT = (
    "Write a single paragraph (3–6 sentences) that would plausibly answer "
    "the question below. Sound confident and factual. It is OK if some "
    "details are guesses — this paragraph is only used to search a "
    "document index, not shown to the user.\n\n"
    "Question: {query}\n\n"
    "Hypothetical answer:"
)


@query_rewriters.register("hyde")
class HyDERewriter(QueryRewriter):
    name = "HyDE (hypothetical answer)"
    description = (
        "Generates a hypothetical answer with the LLM and retrieves using "
        "that. Best for vague or conceptual questions. Adds 1 LLM call."
    )

    def rewrite(self, query: str, llm: "LLMProvider") -> list[str]:
        messages = [{"role": "user", "content": _HYDE_PROMPT.format(query=query)}]
        hypothetical = "".join(llm.stream(messages)).strip()
        if not hypothetical:
            return [query]
        return [hypothetical]
