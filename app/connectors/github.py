"""GitHub connector — fetch text/code files from a public repo via the API.

Accepts URLs like:
    https://github.com/owner/repo
    https://github.com/owner/repo/tree/branch
    owner/repo

Uses the unauthenticated REST API (60 req/hr) unless GITHUB_TOKEN is set
in the environment (5000 req/hr). Only fetches files with text/code-like
extensions, up to GITHUB_MAX_FILES (default 100).
"""
from __future__ import annotations

import os
from typing import Iterator
from urllib.parse import urlparse

import httpx

from . import connectors
from .base import Connector, Document


_TEXT_EXTENSIONS = (
    ".md", ".markdown", ".txt", ".rst", ".adoc", ".org",
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".rs", ".go", ".java", ".kt", ".scala", ".clj",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".m", ".swift",
    ".rb", ".php", ".lua", ".pl", ".sh", ".bash", ".zsh", ".fish",
    ".html", ".css", ".scss", ".less",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".sql", ".graphql", ".proto",
    ".dockerfile",
)


@connectors.register("github")
class GitHubConnector(Connector):
    name = "GitHub repo (public)"
    description = (
        "Index text and code files from a public GitHub repo. "
        "Optional GITHUB_TOKEN env var lifts rate-limit to 5000/hr."
    )
    input_kind = "url"

    def __init__(self) -> None:
        self.max_files = int(os.getenv("GITHUB_MAX_FILES", "100"))
        self.token = os.getenv("GITHUB_TOKEN", "").strip()

    def _parse(self, ref: str) -> tuple[str, str]:
        ref = ref.strip()
        if "://" in ref:
            path = urlparse(ref).path.strip("/")
        else:
            path = ref.strip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                "Expected 'owner/repo' or 'https://github.com/owner/repo'."
            )
        owner = parts[0]
        repo = parts[1].removesuffix(".git")
        return owner, repo

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        owner, repo = self._parse(payload)

        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        with httpx.Client(headers=headers, timeout=30.0) as client:
            # HEAD resolves to the default branch automatically.
            tree_url = (
                f"https://api.github.com/repos/{owner}/{repo}"
                "/git/trees/HEAD?recursive=1"
            )
            r = client.get(tree_url)
            r.raise_for_status()
            tree = r.json().get("tree", [])
            if not tree:
                return

            served = 0
            for item in tree:
                if served >= self.max_files:
                    break
                if item.get("type") != "blob":
                    continue
                path = item.get("path", "")
                if not path:
                    continue
                lower = path.lower()
                if not any(lower.endswith(ext) for ext in _TEXT_EXTENSIONS):
                    continue
                # Skip giant blobs (>2 MB) — likely auto-generated.
                if item.get("size", 0) > 2_000_000:
                    continue
                raw_url = (
                    f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
                )
                try:
                    raw = client.get(raw_url)
                    raw.raise_for_status()
                except Exception:
                    continue
                text = raw.text
                if not text.strip():
                    continue
                yield Document(
                    text=text,
                    source=f"github.com/{owner}/{repo}/{path}",
                )
                served += 1
