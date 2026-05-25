"""Excel connector — openpyxl, one Document per sheet."""
from __future__ import annotations

import io
from typing import Iterator

import openpyxl

from . import connectors
from .base import Connector, Document


@connectors.register("xlsx")
class ExcelConnector(Connector):
    name = "Excel (.xlsx)"
    description = "Upload an Excel workbook; one Document per sheet, rows joined with '|'."
    input_kind = "file"
    accepted_extensions = ["xlsx", "xlsm"]

    def fetch(self, payload: bytes, source_name: str) -> Iterator[Document]:
        wb = openpyxl.load_workbook(
            io.BytesIO(payload), data_only=True, read_only=True
        )
        for sheet_idx, sheet in enumerate(wb.worksheets, start=1):
            rows: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(v) for v in row if v not in (None, "")]
                if values:
                    rows.append(" | ".join(values))
            if not rows:
                continue
            text = f"Sheet: {sheet.title}\n\n" + "\n".join(rows)
            yield Document(
                text=text,
                source=f"{source_name}#{sheet.title}",
                page=sheet_idx,
            )
