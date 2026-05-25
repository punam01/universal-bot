"""PDF connector — pypdfium2 per-page text extraction (text-based PDFs)."""
from __future__ import annotations

import io
from typing import Iterator

import pypdfium2 as pdfium

from . import connectors
from .base import Connector, Document


@connectors.register("pdf")
class PDFConnector(Connector):
    name = "PDF (text-based)"
    description = "Upload a text-based PDF; yields one Document per page."
    input_kind = "file"
    accepted_extensions = ["pdf"]

    def fetch(self, payload: bytes, source_name: str) -> Iterator[Document]:
        pdf = pdfium.PdfDocument(io.BytesIO(payload))
        for i in range(len(pdf)):
            text = pdf[i].get_textpage().get_text_range() or ""
            if text.strip():
                yield Document(text=text, source=source_name, page=i + 1)
