# Plugin SDK

universal-bot is built around five plugin slots. Each slot is a Python package under `app/` with the same three-piece skeleton:

```
app/<slot>/
├── __init__.py     # creates the Registry, auto-imports siblings
├── base.py         # the ABC every plugin in this slot extends
├── <impl1>.py      # one plugin per file, decorated with @registry.register("key")
└── <impl2>.py
```

Auto-discovery happens at import time: `__init__.py` iterates its sibling modules and imports each, so the `@register("name")` decorators fire and populate the registry. **No imperative registration list anywhere** — drop a file in the folder, it shows up.

The five slots:

| Slot | Folder | What it does |
|---|---|---|
| Connectors | `app/connectors/` | Adapt a file / URL / API into normalized `Document`s |
| Indexers | `app/indexers/` | Store chunks and retrieve them by query |
| Query rewriters | `app/query_rewriters/` | Transform the user question before retrieval |
| Rerankers | `app/rerankers/` | Reorder candidate chunks by relevance |
| Providers | `app/providers/` | Stream LLM completions for the chat answer |

---

## The shared pattern

```python
# app/<slot>/__init__.py  (already exists; you don't need to touch it)
import importlib
import pkgutil
from core.registry import Registry
from .base import MyPluginType

my_registry: Registry[MyPluginType] = Registry("my_slot")

for _, _name, _ in pkgutil.iter_modules(__path__):
    if not _name.startswith("_") and _name != "base":
        importlib.import_module(f"{__name__}.{_name}")
```

Inside your plugin file:

```python
# app/<slot>/my_thing.py
from . import my_registry
from .base import MyPluginType

@my_registry.register("my_thing")
class MyThing(MyPluginType):
    name = "Display name shown in the UI"
    # ... implement the abstract methods from the base class
```

Every base class also exposes an optional `is_available()` classmethod. Override it to gate the plugin on runtime conditions (required env var present, optional dep installed, etc.) — the UI only shows plugins whose `is_available()` returns truthy.

---

## Slot reference

### Connectors — `app/connectors/`

**Base contract** (`app/connectors/base.py`):

```python
class Connector(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    input_kind: ClassVar[str]   # "file" | "url" | "text"

    @abstractmethod
    def fetch(self, payload, source_name: str) -> Iterator[Document]:
        ...
```

`input_kind` tells the sidebar UI which input widget to render (file uploader vs URL text input). `payload` is whatever that widget yields — `bytes` for files, `str` for URLs.

A `Document` is just:

```python
@dataclass
class Document:
    text: str
    source: str          # filename or URL
    page: int | None = None
```

**Example — see `app/connectors/pdf.py`** (per-page PDF extraction via pypdfium2).

### Indexers — `app/indexers/`

**Base contract** (`app/indexers/base.py`):

```python
class Indexer(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    is_composite: ClassVar[bool] = False
    deps: ClassVar[list[str]] = []     # other indexer keys this one depends on

    def __init__(self, ctx: IndexerContext, deps=None): ...

    @abstractmethod
    def index(self, docs, chunk_size, overlap) -> int: ...
    @abstractmethod
    def retrieve(self, query, top_k=5) -> list[dict]: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def list_sources(self) -> list[dict]: ...
    @abstractmethod
    def delete_source(self, source: str) -> int: ...
```

`IndexerContext` is the dependency-injection bag the engine constructs once and hands to every indexer:

```python
@dataclass
class IndexerContext:
    chroma_client: chromadb.PersistentClient
    embedder: TextEmbedding
    collection_base: str   # use this as a prefix for your collection / file names
    storage_dir: Path
```

Use whatever subset your indexer needs (semantic uses chroma + embedder; syntactic uses storage_dir for a pickle file).

**Composite indexers** (like `hybrid`) wrap other indexers. Declare your dependencies with `deps = [...]` — the engine resolves them recursively, caches the instances, and injects them via the `deps` kwarg. This keeps in-memory state in sync between standalone and composite views.

**`list_sources()` / `delete_source()`** power the sidebar's source management UI. Implement them so users can see and prune what's indexed without nuking everything.

**Example — see `app/indexers/semantic.py`** (ChromaDB) and `app/indexers/syntactic.py` (rank-bm25).

### Query rewriters — `app/query_rewriters/`

**Base contract**:

```python
class QueryRewriter(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""

    @abstractmethod
    def rewrite(self, query: str, llm: LLMProvider) -> list[str]:
        ...
```

The rewriter runs **before retrieval**. Return one or more queries — the engine retrieves for each and merges results (dedupe by `(source, text[:200])`, keep max score). If you return `[original_query]`, you've effectively no-op'd.

**Example — see `app/query_rewriters/hyde.py`** (~30 lines, generates a hypothetical answer and returns it as the new query).

### Rerankers — `app/rerankers/`

**Base contract**:

```python
class Reranker(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        ...
```

`candidates` is whatever the indexer returned (pre-merge if a rewriter was used). Return `top_k` items in your preferred order; each should still have `text`, `source`, `page`, and a `score`.

**Example — see `app/rerankers/flashrank_reranker.py`** (lazy-loads a 4 MB cross-encoder on first call).

### Providers — `app/providers/`

**Base contract**:

```python
class LLMProvider(ABC):
    name: ClassVar[str]
    env_key: ClassVar[str]              # env var holding the API key
    default_model: ClassVar[str]
    env_model: ClassVar[str] = ""       # optional env var to override model

    def __init__(self, api_key: str, model: str | None = None): ...

    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        ...
```

`messages` follows the OpenAI chat shape: a list of `{"role": "system"|"user"|"assistant", "content": "..."}`. If your provider has a different native format (e.g. Gemini has no `system` role), translate inside `stream`.

`is_available()` is overridden in the base class to return `True` only when `env_key` is set — that's why the provider dropdown only shows providers you have keys for.

**Example — see `app/providers/groq.py`** for an OpenAI-compatible provider, or `app/providers/gemini.py` for a non-standard chat shape.

---

## End-to-end walkthrough: add a Mistral provider

This is what a new LLM provider looks like, top to bottom. About 25 lines.

### 1. Install the SDK

```powershell
.\.venv\Scripts\Activate.ps1
pip install mistralai
```

Add it to `app/requirements.txt`:

```
mistralai>=1.0
```

### 2. Drop the plugin file

`app/providers/mistral.py`:

```python
"""Mistral provider — La Plateforme free tier."""
from __future__ import annotations

from typing import Iterator

from mistralai import Mistral

from . import providers
from .base import LLMProvider


@providers.register("mistral")
class MistralProvider(LLMProvider):
    name = "Mistral Small"
    env_key = "MISTRAL_API_KEY"
    env_model = "MISTRAL_MODEL"
    default_model = "mistral-small-latest"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        super().__init__(api_key, model)
        self._client = Mistral(api_key=self.api_key)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        for event in self._client.chat.stream(model=self.model, messages=messages):
            delta = event.data.choices[0].delta.content
            if delta:
                yield delta
```

### 3. Add your key

```
# .env
MISTRAL_API_KEY=your-key-here
```

### 4. Restart Streamlit

```powershell
streamlit run streamlit_app.py
```

**Mistral Small** appears in the LLM provider dropdown. Nothing else changed.

That's the whole pattern. Same shape for every slot.

---

## Tips

- **License check.** The runtime is strictly permissive (MIT / Apache-2.0 / BSD / ISC / PostgreSQL / MPL). Don't introduce AGPL, GPL, or SSPL dependencies. Common traps: `PyMuPDF` (AGPL — use `pypdfium2`), `trafilatura` (GPL — use `beautifulsoup4`), and the `redis` package ≥ v7.4 (SSPL — use `valkey-py`).

- **Env vars over constructor args.** Plugins read their config from `os.getenv(...)` (often via the base class's `from_env()` classmethod). Don't expect the UI to pass arbitrary config — it only knows about the plugin key and the things on the base class.

- **Lazy heavy state.** If your plugin loads a large model or opens a network connection, defer that work to the first call rather than `__init__`. The engine memoizes instances per key, so `__init__` runs once per plugin per process — but it runs on app startup, before the user has done anything.

- **Be idempotent.** Anything destructive (`reset`, `delete_source`, `delete_collection`) should not raise if the underlying state is already gone. The engine assumes idempotency in `reset_all` etc.

- **No imperative registration.** Never write `my_registry.register("foo")(MyClass)` outside a class definition. The decorator pattern is the contract; sticking to it keeps the auto-discovery readable.
