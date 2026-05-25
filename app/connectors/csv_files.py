"""CSV / TSV connector — stdlib csv with delimiter sniffing."""
from __future__ import annotations

import csv
import io
from typing import Iterator

from . import connectors
from .base import Connector, Document


@connectors.register("csv")
class CsvConnector(Connector):
    name = "CSV / TSV"
    description = "Upload a CSV or TSV; one Document with all rows joined with '|'."
    input_kind = "file"
    accepted_extensions = ["csv", "tsv"]

    def fetch(self, payload: bytes, source_name: str) -> Iterator[Document]:
        text = payload.decode("utf-8", errors="replace")

        delimiter = ","
        try:
            sample = text[:4096]
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except Exception:
            pass

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows: list[str] = []
        for row in reader:
            cells = [c.strip() for c in row if c is not None]
            if any(cells):
                rows.append(" | ".join(cells))

        if rows:
            yield Document(text="\n".join(rows), source=source_name)
