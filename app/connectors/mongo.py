"""MongoDB connector — index documents from a collection as JSON text.

Needs MONGO_CONNECTION_STRING (e.g. mongodb://user:pass@host:27017).
Input is 'database.collection'. Optional MONGO_MAX_DOCS env caps the number of
documents indexed (default 500).
"""
from __future__ import annotations

import json
import os
from typing import Iterator

from . import connectors
from .base import Connector, Document


@connectors.register("mongo")
class MongoConnector(Connector):
    name = "MongoDB collection"
    description = (
        "Index documents from a MongoDB collection as JSON text. "
        "Needs MONGO_CONNECTION_STRING."
    )
    input_kind = "text"
    placeholder = "database.collection (e.g. mydb.users)"

    @classmethod
    def is_available(cls) -> bool:
        if not os.getenv("MONGO_CONNECTION_STRING", "").strip():
            return False
        try:
            import pymongo  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self) -> None:
        self.conn_str = os.getenv("MONGO_CONNECTION_STRING", "").strip()
        self.max_docs = int(os.getenv("MONGO_MAX_DOCS", "500"))

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        from pymongo import MongoClient

        path = payload.strip()
        if "." not in path:
            raise ValueError("Expected 'database.collection' format.")
        db_name, coll_name = path.split(".", 1)

        client = MongoClient(self.conn_str, serverSelectionTimeoutMS=10_000)
        try:
            coll = client[db_name][coll_name]
            for doc in coll.find().limit(self.max_docs):
                doc.pop("_id", None)
                text = json.dumps(doc, default=str, ensure_ascii=False, indent=2)
                if text and text != "{}":
                    yield Document(
                        text=text,
                        source=f"mongo:{db_name}.{coll_name}",
                    )
        finally:
            client.close()
