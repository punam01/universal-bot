"""Phase 2.5 UI — adds reranker dropdown + multi-turn chat memory."""
from __future__ import annotations

import streamlit as st

from config import settings
from rag import RAGEngine


st.set_page_config(page_title="universal-bot", layout="wide")
st.title("universal-bot")
st.caption(
    "Pluggable connectors, indexers, rerankers, and LLM providers. "
    "Multi-turn chat with cited sources."
)


@st.cache_resource(
    show_spinner="Starting engine (downloads embedding model on first run)..."
)
def get_engine() -> RAGEngine:
    return RAGEngine()


engine = get_engine()


def _render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            page = s.get("page")
            page_label = f" — page {page}" if page else ""
            st.caption(
                f"**[{i}] {s['source']}**{page_label} "
                f"(score {s['score']:.3f})"
            )
            preview = s["text"]
            if len(preview) > 500:
                preview = preview[:500] + "..."
            st.text(preview)


def _default_index(options: list[tuple[str, str]], desired: str) -> int:
    keys = [k for k, _ in options]
    return keys.index(desired) if desired in keys else 0


# ---------- Sidebar: configuration + ingest ----------
with st.sidebar:
    st.header("Configuration")

    provider_options = engine.available_providers()
    if not provider_options:
        st.error(
            "No LLM provider keys found in .env. Set at least one of "
            "GROQ_API_KEY or GOOGLE_API_KEY."
        )
        st.stop()
    provider_key = st.selectbox(
        "LLM provider",
        options=[k for k, _ in provider_options],
        index=_default_index(provider_options, settings.default_provider),
        format_func=lambda k: dict(provider_options)[k],
    )

    indexer_options = engine.available_indexers()
    indexer_key = st.selectbox(
        "Indexer (retrieval strategy)",
        options=[k for k, _ in indexer_options],
        index=_default_index(indexer_options, settings.default_indexer),
        format_func=lambda k: dict(indexer_options)[k],
        help=(
            "`semantic` = vector similarity, `syntactic` = BM25 keyword, "
            "`hybrid` = both fused via RRF."
        ),
    )

    reranker_options = engine.available_rerankers()
    reranker_choices = [("none", "None")] + reranker_options
    reranker_key = st.selectbox(
        "Reranker (optional)",
        options=[k for k, _ in reranker_choices],
        index=_default_index(reranker_choices, settings.default_reranker),
        format_func=lambda k: dict(reranker_choices)[k],
        help=(
            "Reorders the top retrieval results with a small CPU model. "
            "Big quality lift for negligible cost. First use downloads ~4 MB."
        ),
    )

    st.divider()
    st.subheader("Add knowledge")

    connector_options = engine.available_connectors()
    connector_key = st.selectbox(
        "Source type",
        options=[k for k, _ in connector_options],
        index=_default_index(connector_options, settings.default_connector),
        format_func=lambda k: dict(connector_options)[k],
    )

    connector_kind = engine.connector_kind(connector_key)
    connector_name = dict(connector_options)[connector_key]

    if connector_kind == "file":
        pdf = st.file_uploader(f"Upload file ({connector_name})", type=["pdf"])
        if pdf is not None and st.button(
            f"Index {pdf.name}", use_container_width=True
        ):
            with st.spinner("Ingesting..."):
                try:
                    n = engine.ingest(
                        connector_key, indexer_key, pdf.getvalue(), pdf.name
                    )
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")
                else:
                    st.success(f"Indexed {n} chunks into '{indexer_key}'")
    elif connector_kind == "url":
        url = st.text_input("URL", placeholder="https://...")
        if url and st.button(
            f"Run {connector_name}", use_container_width=True
        ):
            with st.spinner(
                f"{connector_name} running... (crawls may take a few minutes)"
            ):
                try:
                    n = engine.ingest(connector_key, indexer_key, url, url)
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")
                else:
                    if n == 0:
                        st.warning("No content extracted.")
                    else:
                        st.success(f"Indexed {n} chunks into '{indexer_key}'")
    else:
        st.info(
            f"No UI yet for input_kind={connector_kind!r}. Drop a matching "
            f"file under app/connectors/ to extend."
        )

    st.divider()
    if st.button("Reset all indexes", use_container_width=True):
        engine.reset_all()
        st.session_state.pop("messages", None)
        st.success("All indexes cleared")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.success("Chat cleared")


# ---------- Main: chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            sources = engine.retrieve(
                indexer_key, prompt, reranker_key=reranker_key
            )
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            sources = []

        # History excludes the user message we just appended; chat_stream
        # appends it back as the final user turn alongside the context.
        history = st.session_state.messages[:-1]

        placeholder = st.empty()
        full = ""
        try:
            for token in engine.chat_stream(
                provider_key, prompt, sources, history=history
            ):
                full += token
                placeholder.markdown(full + "▌")
        except Exception as exc:
            full = f"_LLM error: {exc}_"
        placeholder.markdown(full)
        if sources:
            _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": full, "sources": sources}
    )
