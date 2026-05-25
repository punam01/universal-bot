"""Plain text / Markdown connector."""
from __future__ import annotations

from typing import Iterator

from . import connectors
from .base import Connector, Document


@connectors.register("text")
class TextConnector(Connector):
    name = "Plain text / Markdown"
    description = "Upload a .txt, .md, or .markdown file."
    input_kind = "file"
    accepted_extensions = ["txt", "md", "markdown", "text"]

    def fetch(self, payload: bytes, source_name: str) -> Iterator[Document]:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = payload.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = payload.decode("utf-8", errors="replace")

        if text.strip():
            yield Document(text=text, source=source_name)
