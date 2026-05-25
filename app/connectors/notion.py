"""Notion connector — index a page or database via the Notion REST API.

Needs NOTION_TOKEN (an Internal Integration token). The integration must be
shared with the page/database you want to index. The simplified Notion ID
extractor accepts a full Notion URL or a raw 32-char hex ID.
"""
from __future__ import annotations

import os
import re
from typing import Iterator

import httpx

from . import connectors
from .base import Connector, Document


_ID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|[0-9a-fA-F]{32})"
)


@connectors.register("notion")
class NotionConnector(Connector):
    name = "Notion (page or database)"
    description = "Index Notion blocks. Needs NOTION_TOKEN env + integration share."
    input_kind = "text"
    placeholder = "Notion page URL or 32-char ID"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.getenv("NOTION_TOKEN", "").strip())

    def __init__(self) -> None:
        self.token = os.getenv("NOTION_TOKEN", "").strip()

    @staticmethod
    def _extract_id(ref: str) -> str:
        m = _ID_RE.search(ref)
        if not m:
            raise ValueError(
                "Could not find a Notion ID in input. "
                "Paste a Notion page URL or a 32-char ID."
            )
        return m.group(1).replace("-", "")

    @staticmethod
    def _block_text(block: dict) -> str:
        btype = block.get("type")
        if not btype:
            return ""
        rich = block.get(btype, {}).get("rich_text") or []
        return "".join(rt.get("plain_text", "") for rt in rich).strip()

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        page_id = self._extract_id(payload)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": "2022-06-28",
        }
        with httpx.Client(headers=headers, timeout=30.0) as client:
            blocks: list[dict] = []
            next_cursor: str | None = None
            url = f"https://api.notion.com/v1/blocks/{page_id}/children"
            while True:
                params: dict = {"page_size": 100}
                if next_cursor:
                    params["start_cursor"] = next_cursor
                r = client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                blocks.extend(data.get("results", []))
                if not data.get("has_more"):
                    break
                next_cursor = data.get("next_cursor")

        parts: list[str] = []
        for b in blocks:
            t = self._block_text(b)
            if t:
                parts.append(t)
        text = "\n\n".join(parts).strip()
        if text:
            yield Document(text=text, source=f"notion:{page_id}")
