"""Site-crawl connector — BFS within the seed URL's domain.

Behavior:
- Crawl starts at the URL the user pastes.
- Only follows links on the same registered domain.
- Respects robots.txt (best-effort; ignored if unreachable).
- Bounded by depth, page count, per-request timeout, and a polite delay.

Tuning via env vars (all optional):
- CRAWL_MAX_PAGES   (default 25)
- CRAWL_MAX_DEPTH   (default 2)
- CRAWL_DELAY       (default 0.5  — seconds between requests)
- CRAWL_TIMEOUT     (default 10   — seconds per request)
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Iterator
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from . import connectors
from .base import Connector, Document


_NOISE_TAGS = (
    "script", "style", "noscript", "iframe",
    "header", "footer", "nav", "aside",
)
_HTML_TYPES = ("text/html", "application/xhtml")
_SKIP_LINK_PREFIXES = ("#", "mailto:", "javascript:", "tel:", "data:")


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(_SKIP_LINK_PREFIXES):
            continue
        absolute = urljoin(base_url, href).split("#")[0]
        out.append(absolute)
    return out


def _same_domain(url: str, root_netloc: str) -> bool:
    return urlparse(url).netloc == root_netloc


@connectors.register("crawl")
class SiteCrawlConnector(Connector):
    name = "Whole site (BFS crawl)"
    description = (
        "Crawls every same-domain page reachable from the seed URL, "
        "bounded by depth/page limits and robots.txt."
    )
    input_kind = "url"

    def __init__(self) -> None:
        self.max_pages = int(os.getenv("CRAWL_MAX_PAGES", "25"))
        self.max_depth = int(os.getenv("CRAWL_MAX_DEPTH", "2"))
        self.request_delay = float(os.getenv("CRAWL_DELAY", "0.5"))
        self.timeout = float(os.getenv("CRAWL_TIMEOUT", "10"))
        self._ua = "universal-bot/0.2 (+plug-and-play RAG)"

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        seed = payload.strip()
        seed_parsed = urlparse(seed)
        if not seed_parsed.scheme or not seed_parsed.netloc:
            raise ValueError(f"Invalid URL: {seed!r}")
        root_netloc = seed_parsed.netloc
        scheme = seed_parsed.scheme

        # robots.txt — fail-open if unreachable
        robots: RobotFileParser | None = RobotFileParser()
        robots.set_url(f"{scheme}://{root_netloc}/robots.txt")
        try:
            robots.read()
        except Exception:
            robots = None

        def allowed(url: str) -> bool:
            if robots is None:
                return True
            try:
                return robots.can_fetch(self._ua, url)
            except Exception:
                return True

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(seed, 0)])

        with httpx.Client(
            follow_redirects=True,
            timeout=self.timeout,
            headers={"User-Agent": self._ua},
        ) as client:
            while queue and len(visited) < self.max_pages:
                url, depth = queue.popleft()
                if url in visited:
                    continue
                if not _same_domain(url, root_netloc):
                    continue
                if not allowed(url):
                    continue
                visited.add(url)

                try:
                    response = client.get(url)
                    response.raise_for_status()
                except Exception:
                    continue

                ctype = response.headers.get("content-type", "").lower()
                if not any(c in ctype for c in _HTML_TYPES):
                    continue

                html = response.text
                text = _clean_html(html)
                if text:
                    yield Document(text=text, source=url, page=None)

                if depth < self.max_depth:
                    for link in _extract_links(html, url):
                        if link not in visited and _same_domain(link, root_netloc):
                            queue.append((link, depth + 1))

                time.sleep(self.request_delay)
