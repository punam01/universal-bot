"""OCR-PDF connector — render each page as an image and OCR with RapidOCR.

Slower than the regular `pdf` connector (~1–3s per page on CPU) but works on
scanned PDFs where text extraction returns nothing.
"""
from __future__ import annotations

import io
from typing import Iterator

import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR

from . import connectors
from .base import Connector, Document


@connectors.register("ocr_pdf")
class OcrPdfConnector(Connector):
    name = "Scanned PDF (OCR)"
    description = (
        "Use for PDFs that the text-based PDF connector returns 0 chunks for. "
        "Slow (~1–3s per page) — RapidOCR runs locally on CPU. "
        "First use downloads ~10 MB of OCR models."
    )
    input_kind = "file"
    accepted_extensions = ["pdf"]

    def __init__(self) -> None:
        self._ocr: RapidOCR | None = None

    def _get_ocr(self) -> RapidOCR:
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    def fetch(self, payload: bytes, source_name: str) -> Iterator[Document]:
        pdf = pdfium.PdfDocument(io.BytesIO(payload))
        ocr = self._get_ocr()
        for i in range(len(pdf)):
            page = pdf[i]
            # Render at scale=2.0 (~144 DPI) — good OCR quality, manageable size.
            pil = page.render(scale=2.0).to_pil()
            arr = np.array(pil)
            try:
                result, _elapsed = ocr(arr)
            except Exception:
                continue
            if not result:
                continue
            lines = [line[1] for line in result if line and len(line) >= 2 and line[1]]
            text = "\n".join(lines).strip()
            if text:
                yield Document(text=text, source=source_name, page=i + 1)
