"""URL connector — fetches a web page and strips boilerplate."""
from __future__ import annotations

from typing import Iterator

import httpx
from bs4 import BeautifulSoup

from . import connectors
from .base import Connector, Document


_NOISE_TAGS = ("script", "style", "noscript", "iframe", "header", "footer", "nav", "aside")


@connectors.register("url")
class URLConnector(Connector):
    name = "Web page (URL)"
    description = "Fetches the page and indexes its main text content."
    input_kind = "url"

    def fetch(self, payload: str, source_name: str | None = None) -> Iterator[Document]:
        url = payload.strip()
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(
                url, headers={"User-Agent": "universal-bot/0.2"}
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if text:
            yield Document(text=text, source=source_name or url, page=None)
