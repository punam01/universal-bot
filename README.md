# universal-bot

> Universal, plug-and-play RAG chatbot platform. Bring your own docs, bring your own key, bring your own model. Zero-budget by default.

**Status**: Phase 1 — first runnable slice. `docker compose up` → upload a PDF → chat.

## Why universal-bot

Most RAG platforms lock you into a vendor, a model, or both. universal-bot is different:

- **Permissively licensed** — Apache-2.0, no AGPL / SSPL / GPL in the runtime path. Ship it however you want.
- **100% self-hostable** — runs end-to-end on a free Oracle Cloud ARM VM. No SaaS in the critical path.
- **Plug-and-play plugins** — add a data source, indexing strategy, or LLM provider by dropping a single file in the right folder. The UI picks it up automatically.
- **Bring your own key** — free local models via Ollama by default; plug in OpenAI / Anthropic / Gemini / Groq per project.
- **Indexing as a conversation** — an onboarding agent interviews you about your corpus and picks the best strategy (semantic / keyword / hybrid / hierarchical).

## Quickstart (Phase 1)

PDF upload → semantic indexing (local CPU) → cited chat answers (free Gemini 2.0 Flash). Runs natively on Windows — no Docker, no virtualization, no admin needed.

### Prerequisites
- **Python 3.11+** — `winget install Python.Python.3.11` or grab the installer from https://www.python.org/downloads/ (tick "Add to PATH")
- A free **Gemini API key** — https://aistudio.google.com/app/apikey

### Run (PowerShell)

1. Set your key in `.env` at the repo root:
   ```powershell
   Copy-Item .env.example .env -ErrorAction SilentlyContinue
   notepad .env
   ```
   Set `GOOGLE_API_KEY=your-actual-key`, save, close.

2. Install dependencies (one-time, ~2 min):
   ```powershell
   cd app
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Launch:
   ```powershell
   streamlit run streamlit_app.py
   ```

Open http://localhost:8501 — sidebar → upload PDF → click **Index** → ask questions.

First indexing call downloads the bge-small embedding model (~133 MB) into `~/.cache/fastembed`; subsequent runs are instant.

### Stop / clean
- Stop: `Ctrl+C` in the terminal
- Wipe vector index: delete `app/chroma_data/`
- Reinstall deps from scratch: delete `app/.venv` and repeat step 2

### Phase 1 stack
| Component | Where it runs |
|---|---|
| Streamlit UI | localhost:8501 |
| ChromaDB | embedded in the same Python process (no separate server) |
| Embeddings | fastembed (CPU, local) |
| LLM | Gemini 2.0 Flash (free API) |

> `docker-compose.yml` and `app/Dockerfile` are kept in the repo for the day Docker becomes available on this machine. They are not used in the native Phase 1 path.

## Roadmap

- [x] **Phase 0** — scaffolding, free infra provisioning
- [ ] **Phase 1** — thin vertical slice: PDF upload → semantic index → chat
- [ ] **Phase 2** — plugin registries (connectors, indexers, providers)
- [ ] **Phase 3** — LangGraph onboarding agent + hierarchical/hybrid indexers
- [ ] **Phase 4** — BYOK with SOPS + multi-provider support
- [ ] **Phase 5** — OpenLLMetry observability + Ragas CI
- [ ] **Phase 6** — docs site + public launch

## Stack (permissive licenses only)

| Layer | Tool | License |
|---|---|---|
| API | FastAPI | MIT |
| Frontend | Next.js + shadcn/ui | MIT |
| RAG | LlamaIndex + LangGraph | MIT |
| Vector DB | Qdrant | Apache-2.0 |
| Relational | PostgreSQL + pgvector | PostgreSQL |
| Cache / queue | Valkey | BSD-3 |
| Object store | Local FS → SeaweedFS | Apache-2.0 |
| LLM proxy | LiteLLM | MIT |
| Local LLM | Ollama | MIT |
| Embeddings | fastembed | Apache-2.0 |
| Reranker | FlashRank | Apache-2.0 |
| Parsing | Docling + pypdfium2 | MIT / Apache-2.0 |
| Secrets | SOPS + age | MPL-2.0 / BSD-3 |
| Observability | OpenLLMetry + Jaeger | Apache-2.0 |

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All commits must be signed off (`git commit -s`) per the [DCO](https://developercertificate.org/).

## Security

See [SECURITY.md](SECURITY.md).
