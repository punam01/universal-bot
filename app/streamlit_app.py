"""Phase 1 UI — minimal Streamlit chat with PDF upload."""
from __future__ import annotations

import streamlit as st

from rag import RAGEngine


st.set_page_config(page_title="universal-bot — Phase 1", layout="wide")
st.title("universal-bot")
st.caption(
    "Phase 1 — upload a PDF, ask questions. "
    "Free Gemini 2.0 Flash + local CPU embeddings (bge-small)."
)


@st.cache_resource(show_spinner="Starting engine (downloads embedding model on first run)...")
def get_engine() -> RAGEngine:
    return RAGEngine()


def _render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.caption(
                f"**[{i}] {s['source']}** — page {s.get('page', '?')} "
                f"(score {s['score']:.3f})"
            )
            preview = s["text"]
            if len(preview) > 500:
                preview = preview[:500] + "..."
            st.text(preview)


engine = get_engine()


# ---------- Sidebar: knowledge management ----------
with st.sidebar:
    st.header("Knowledge")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    if pdf is not None:
        if st.button(f"Index {pdf.name}", use_container_width=True):
            with st.spinner("Extracting → chunking → embedding..."):
                n = engine.ingest_pdf(pdf.getvalue(), pdf.name)
            if n:
                st.success(f"Indexed {n} chunks from {pdf.name}")
            else:
                st.warning(
                    "No text extracted. Is this a scanned/image-only PDF? "
                    "OCR support arrives in a later phase."
                )

    st.divider()
    if st.button("Reset index", use_container_width=True, type="secondary"):
        engine.reset()
        st.session_state.pop("messages", None)
        st.success("Index cleared")


# ---------- Main: chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])

# New turn
if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources = engine.retrieve(prompt)
        placeholder = st.empty()
        full = ""
        for token in engine.chat_stream(prompt, sources):
            full += token
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)
        if sources:
            _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": full, "sources": sources}
    )
