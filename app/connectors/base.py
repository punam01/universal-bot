"""Connector plugin contract — adapts external data into normalized Documents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Iterator


@dataclass
class Document:
    text: str
    source: str
    page: int | None = None


class Connector(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    input_kind: ClassVar[str]                       # "file" | "url" | "text"
    accepted_extensions: ClassVar[list[str]] = []   # extensions for input_kind="file"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abstractmethod
    def fetch(self, payload, source_name: str) -> Iterator[Document]:
        """Yield Documents extracted from the input payload."""
