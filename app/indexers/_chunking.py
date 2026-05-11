"""Shared chunking helpers used by all indexers."""
from __future__ import annotations


def sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of `size` characters."""
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
