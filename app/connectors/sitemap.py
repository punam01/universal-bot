"""Sitemap connector — fetch every URL listed in a sitemap.xml.

Same-domain only. Supports flat sitemaps and one level of nested sitemap
indexes (a sitemap that lists other sitemaps).
"""
from __future__ import annotations

import os
import time
from typing import Iterator
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from . import connectors
from .base import Connector, Document


_NOISE_TAGS = (
    "script", "style", "noscript", "iframe",
    "header", "footer", "nav", "aside",
)


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


@connectors.register("sitemap")
class SitemapConnector(Connector):
    name = "Sitemap (XML)"
    description = (
        "Fetch every URL listed in a sitemap.xml. Same-domain only. "
        "Supports nested sitemap indexes (one level deep)."
    )
    input_kind = "url"

    def __init__(self) -> None:
        self.max_pages = int(os.getenv("SITEMAP_MAX_PAGES", "100"))
        self.request_delay = float(os.getenv("SITEMAP_DELAY", "0.3"))
        self.timeout = float(os.getenv("SITEMAP_TIMEOUT", "10"))

    def _list_urls(self, client: httpx.Client, sitemap_url: str) -> list[str]:
        resp = client.get(sitemap_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
        # Nested sitemap index
        nested = [loc.text.strip() for loc in soup.find_all("sitemap")]
        if nested:
            urls: list[str] = []
            for sub in nested:
                try:
                    sub_resp = client.get(sub)
                    sub_resp.raise_for_status()
                except Exception:
                    continue
                sub_soup = BeautifulSoup(sub_resp.text, "xml")
                urls.extend(loc.text.strip() for loc in sub_soup.find_all("loc"))
            return urls
        return [loc.text.strip() for loc in soup.find_all("loc")]

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        seed = payload.strip()
        root_netloc = urlparse(seed).netloc
        if not root_netloc:
            raise ValueError(f"Invalid sitemap URL: {seed!r}")

        with httpx.Client(
            follow_redirects=True,
            timeout=self.timeout,
            headers={"User-Agent": "universal-bot/0.3"},
        ) as client:
            urls = self._list_urls(client, seed)

            for url in urls[: self.max_pages]:
                if urlparse(url).netloc != root_netloc:
                    continue
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                except Exception:
                    continue
                if "html" not in resp.headers.get("content-type", "").lower():
                    continue
                text = _clean_html(resp.text)
                if text:
                    yield Document(text=text, source=url)
                time.sleep(self.request_delay)
