"""SQL connector — run a SELECT and index each row.

Set SQL_CONNECTION_STRING in .env to a SQLAlchemy URL, e.g.
- sqlite:///path/to/file.db
- postgresql+psycopg://user:pass@host/db   (needs `pip install psycopg[binary]`)
- mysql+pymysql://user:pass@host/db        (needs `pip install pymysql`)

Only SELECT statements are accepted — the connector explicitly rejects
anything else to guard against accidental writes.
"""
from __future__ import annotations

import os
from typing import Iterator

from . import connectors
from .base import Connector, Document


_WRITE_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "grant", "revoke", "merge", "replace",
)


@connectors.register("sql")
class SqlConnector(Connector):
    name = "SQL database (SELECT)"
    description = (
        "Run a SELECT against any SQLAlchemy-supported DB. "
        "Needs SQL_CONNECTION_STRING; install your dialect's driver separately."
    )
    input_kind = "text"
    placeholder = "SELECT id, title, body FROM articles LIMIT 200"

    @classmethod
    def is_available(cls) -> bool:
        if not os.getenv("SQL_CONNECTION_STRING", "").strip():
            return False
        try:
            import sqlalchemy  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self) -> None:
        self.conn_str = os.getenv("SQL_CONNECTION_STRING", "").strip()
        self.max_rows = int(os.getenv("SQL_MAX_ROWS", "1000"))

    @staticmethod
    def _validate(query: str) -> None:
        stripped = query.strip().lower()
        if not stripped.startswith("select"):
            raise ValueError("Only SELECT statements are accepted.")
        # Reject any write keyword in any subquery
        first_token_after_select = stripped.split()[0:5]
        for word in _WRITE_KEYWORDS:
            if word in stripped.split():
                raise ValueError(f"Rejected: keyword '{word}' found in query.")

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        from sqlalchemy import create_engine, text as sql_text

        query = payload.strip()
        self._validate(query)

        engine = create_engine(self.conn_str)
        with engine.connect() as conn:
            result = conn.execute(sql_text(query))
            columns = list(result.keys())
            label = f"sql:{query[:60]}"
            count = 0
            for row in result:
                if count >= self.max_rows:
                    break
                row_text = " | ".join(
                    f"{col}: {val}"
                    for col, val in zip(columns, row)
                    if val is not None
                )
                if row_text:
                    yield Document(text=row_text, source=label)
                    count += 1
