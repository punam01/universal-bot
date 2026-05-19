# universal-bot

> Plug-and-play RAG chatbot platform. Drop a file in `app/connectors/`, `app/indexers/`, `app/providers/`, `app/rerankers/`, or `app/query_rewriters/` — it shows up in the UI on the next restart.

**Status**: usable (not 1.0). 5 plugin slots, 12 plugins shipped, multi-turn chat, source management, corpus advisor. Runs natively on Python with no Docker required.

## What you get

- **Native Python**, embedded ChromaDB — no Docker, no separate vector DB server
- **3 connectors** (PDF, single URL, full-site BFS crawler)
- **3 indexers** (semantic vector, syntactic BM25, hybrid via Reciprocal Rank Fusion)
- **2 query rewriters** (HyDE, multi-query expansion)
- **1 reranker** (FlashRank — tiny CPU model, big quality lift)
- **2 LLM providers** (Groq Llama 3.3 70B, Gemini 2.0 Flash) — both BYOK, both free-tier
- **Multi-turn chat** with cited sources
- **Source management** — list, per-source delete
- **Corpus advisor** — single LLM call that samples your content and recommends a pipeline

Permissive-licensed runtime (MIT / Apache-2.0 / BSD / ISC / PostgreSQL / MPL). No AGPL, GPL, or SSPL.

## Quickstart

### Prerequisites
- **Python 3.11+** (3.14 is currently too new for some wheels — install 3.11 if you only have 3.14: `winget install Python.Python.3.11`)
- A free API key for at least one provider:
  - **Groq** (recommended — generous free tier): https://console.groq.com/keys
  - **Gemini**: https://aistudio.google.com/app/apikey

### Run

```powershell
# 1. Config
Copy-Item .env.example .env -ErrorAction SilentlyContinue
notepad .env
# Set GROQ_API_KEY=... (and/or GOOGLE_API_KEY=...). Save, close.

# 2. Install (one time, ~2 min)
cd app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Launch
streamlit run streamlit_app.py
```

Browser opens at `http://localhost:8501`. Sidebar → upload a PDF or paste a URL → click **Index** → ask questions in the chat.

First indexing call downloads the bge-small embedding model (~133 MB) into `~/.cache/fastembed`. Subsequent runs are instant. The FlashRank reranker downloads ~4 MB on first use.

## Architecture

A query flows through up to 4 stages:

```
question
    │
    ▼
[ query rewriter ]   ← optional. hyde / multi_query (1 LLM call)
    │
    ▼ (one or more queries)
[ indexer ]          ← semantic / syntactic / hybrid
    │
    ▼ (top-k chunks)
[ reranker ]         ← optional. flashrank (1 CPU pass, ~50ms)
    │
    ▼ (best-k chunks)
[ provider ]         ← groq / gemini
    │
    ▼
streamed answer with [1] [2] citations
```

Ingest is symmetric:

```
file/url → [ connector ] → Documents → [ indexer ] → persisted chunks
```

Each stage is a registered plugin discovered at import time. The orchestrator in `app/rag.py` knows the registries — not the plugin internals.

## Add a plugin

The plugin SDK is just a `Registry` + an `@register("name")` decorator + an auto-discovery import in `__init__.py`. Concrete authoring guide with copy-paste templates: **[docs/plugin-sdk.md](docs/plugin-sdk.md)**.

Example: an entire new LLM provider in 20 lines (drop in `app/providers/mistral.py`):

```python
from mistralai import Mistral
from . import providers
from .base import LLMProvider

@providers.register("mistral")
class MistralProvider(LLMProvider):
    name = "Mistral"
    env_key = "MISTRAL_API_KEY"
    default_model = "mistral-small-latest"

    def __init__(self, api_key, model=None):
        super().__init__(api_key, model)
        self._client = Mistral(api_key=api_key)

    def stream(self, messages):
        for chunk in self._client.chat.stream(model=self.model, messages=messages):
            if chunk.data.choices[0].delta.content:
                yield chunk.data.choices[0].delta.content
```

Restart Streamlit. **Mistral** appears in the provider dropdown. No other code change.

## Current stack (actual, not aspirational)

| Layer | Tool | License |
|---|---|---|
| UI | Streamlit | Apache-2.0 |
| Vector DB | ChromaDB (embedded) | Apache-2.0 |
| Embeddings | fastembed (`bge-small-en-v1.5`, CPU/ONNX) | Apache-2.0 |
| Keyword index | rank-bm25 (in-process) | Apache-2.0 |
| PDF parse | pypdfium2 | Apache-2.0 / BSD-3 |
| HTML fetch / parse | httpx + beautifulsoup4 | BSD-3 / MIT |
| Reranker | FlashRank (`ms-marco-MiniLM-L-12-v2`) | Apache-2.0 |
| LLM SDKs | groq, google-genai | Apache-2.0 |
| Config | python-dotenv | BSD-3 |

`docker-compose.yml` and `app/Dockerfile` are kept in the repo for future containerized deployment — they're not used by the native quickstart.

## Roadmap

- [x] **Phase 0** — scaffolding, license/CI/DCO setup
- [x] **Phase 1** — PDF → semantic → cited chat (Streamlit + ChromaDB + Groq/Gemini)
- [x] **Phase 2** — plugin registries for connectors, indexers, providers; crawl + hybrid + reranker + multi-turn + source management
- [x] **Phase 3** — corpus advisor recommends pipeline settings
- [ ] **Phase 4** — multiple isolated projects; encrypted BYOK; OCR for scanned PDFs
- [ ] **Phase 5** — Ragas eval gate in CI; per-call traces (OpenLLMetry → Jaeger)
- [ ] **Phase 6** — MkDocs site; public launch

## Project layout

```
app/
├── connectors/        # pdf.py, url.py, crawl.py + base + registry
├── indexers/          # semantic.py, syntactic.py, hybrid.py + base + registry
├── query_rewriters/   # hyde.py, multi_query.py + base + registry
├── rerankers/         # flashrank_reranker.py + base + registry
├── providers/         # groq.py, gemini.py + base + registry
├── core/registry.py   # the shared Registry[T] pattern
├── advisor.py         # corpus → pipeline recommendation
├── rag.py             # orchestrator (no plugin internals)
├── streamlit_app.py   # UI
└── config.py          # env-driven defaults
```

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All commits must be signed off (`git commit -s`) per the [DCO](https://developercertificate.org/). To add a plugin, see [docs/plugin-sdk.md](docs/plugin-sdk.md).

## Security

See [SECURITY.md](SECURITY.md).
