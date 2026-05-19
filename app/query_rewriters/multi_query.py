"""Multi-query expansion — paraphrase the question N ways, retrieve for each, merge.

Better recall on queries where wording matters (synonyms, terminology
differences between user and source). Costs 1 LLM call.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from . import query_rewriters
from .base import QueryRewriter

if TYPE_CHECKING:
    from providers.base import LLMProvider


_PROMPT = (
    "Rephrase the following question into {n} different ways. "
    "Each rephrasing should ask the SAME thing but use different wording, "
    "synonyms, or angle. Output exactly one rephrasing per line, no "
    "numbering, no extra commentary.\n\n"
    "Original: {query}\n\nRephrasings:"
)

_NUMBER_PREFIX = re.compile(r"^\s*(?:\d+[.)\]]|[-*])\s*")


@query_rewriters.register("multi_query")
class MultiQueryRewriter(QueryRewriter):
    name = "Multi-query (3 paraphrases + original)"
    description = (
        "Paraphrases the query 3 ways and retrieves for each, then dedupes "
        "results. Improves recall on terminology-heavy corpora. Adds 1 LLM call."
    )
    n_variants: ClassVar[int] = 3

    def rewrite(self, query: str, llm: "LLMProvider") -> list[str]:
        messages = [
            {
                "role": "user",
                "content": _PROMPT.format(n=self.n_variants, query=query),
            }
        ]
        raw = "".join(llm.stream(messages))
        variants: list[str] = []
        for line in raw.splitlines():
            stripped = _NUMBER_PREFIX.sub("", line).strip()
            if stripped:
                variants.append(stripped)
        # Always include the original, then the variants up to n.
        return [query] + variants[: self.n_variants]
