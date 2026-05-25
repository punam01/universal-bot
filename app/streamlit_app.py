"""Streamlit UI — projects + plugin pickers + sources + advisor + eval + chat."""
from __future__ import annotations

import streamlit as st

from config import settings
from rag import RAGEngine


st.set_page_config(page_title="universal-bot", layout="wide")
st.title("universal-bot")
st.caption(
    "Multi-project plug-and-play RAG. Switch projects to keep knowledge bases "
    "isolated. Pluggable connectors, indexers, rewriters, rerankers, providers."
)


@st.cache_resource(
    show_spinner="Starting engine (downloads embedding model on first run)..."
)
def get_engine() -> RAGEngine:
    return RAGEngine()


engine = get_engine()

# Apply pending project switch from previous rerun
pending_project = st.session_state.pop("_pending_project", None)
if pending_project is not None:
    try:
        engine.set_project(pending_project)
        st.session_state.pop("messages", None)
        st.session_state.pop("advisor_result", None)
        st.session_state.pop("eval_result", None)
    except ValueError as exc:
        st.error(str(exc))


# ---------- helpers ----------

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
    if desired in keys:
        return keys.index(desired)
    return 0


def _short(name: str, limit: int = 42) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _initial_for(setting_key: str, fallback: str) -> str:
    override = st.session_state.get(setting_key)
    return override if override else fallback


def _friendly_error(exc: Exception) -> str:
    """Render an exception as one tidy line (class + message), no stack noise."""
    msg = str(exc).strip()
    if not msg:
        return type(exc).__name__
    if len(msg) > 240:
        msg = msg[:237] + "…"
    return f"{type(exc).__name__}: {msg}"


def _render_save_eval_button(msg_idx: int, msg: dict, project: str) -> None:
    """Show a 'Save this Q to eval set' button below an assistant message with sources."""
    if msg.get("role") != "assistant" or not msg.get("sources"):
        return
    msgs = st.session_state.messages
    if msg_idx == 0 or msgs[msg_idx - 1].get("role") != "user":
        return
    user_q = msgs[msg_idx - 1].get("content", "")
    expected = sorted({s["source"] for s in msg["sources"]})
    key = f"save_eval::{project}::{msg_idx}::{hash(user_q)}"
    if st.button("Save this Q to eval set", key=key):
        from eval_harness import GoldenQuestion, save_question

        save_question(
            project,
            GoldenQuestion(question=user_q, expected_sources=expected),
        )
        st.toast(
            f"Saved to eval set ({len(expected)} expected source"
            f"{'s' if len(expected) != 1 else ''})."
        )


# ---------- Sidebar ----------
with st.sidebar:
    # ----- Project selector -----
    st.header("Project")
    projects = engine.list_projects()
    current = engine.current_project
    project_choice = st.selectbox(
        "Active project",
        options=projects,
        index=projects.index(current) if current in projects else 0,
        key="ub_project_select",
    )
    if project_choice != current:
        st.session_state["_pending_project"] = project_choice
        st.rerun()

    with st.expander("Manage projects", expanded=False):
        new_name = st.text_input(
            "New project name",
            placeholder="e.g. work-docs, recipes",
            key="ub_new_project_name",
        )
        if st.button("Create project", use_container_width=True):
            try:
                created = engine.create_project(new_name)
                st.session_state["_pending_project"] = created
                st.success(f"Created '{created}'. Switching…")
                st.rerun()
            except ValueError as exc:
                st.error(_friendly_error(exc))

        if current != "default":
            st.divider()
            st.caption(f"Currently on **{current}**.")
            confirm = st.checkbox(
                f"I understand this permanently deletes all data in '{current}'.",
                key="ub_confirm_delete",
            )
            if st.button(
                f"Delete project '{current}'",
                use_container_width=True,
                disabled=not confirm,
            ):
                try:
                    engine.delete_project(current)
                    st.session_state["_pending_project"] = "default"
                    st.success(f"Deleted '{current}'. Back on 'default'.")
                    st.rerun()
                except ValueError as exc:
                    st.error(_friendly_error(exc))
        else:
            st.caption("(switch to a non-default project to enable delete)")

    st.divider()

    # ----- Configuration -----
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
        index=_default_index(provider_options, _initial_for("ub_provider", settings.default_provider)),
        format_func=lambda k: dict(provider_options)[k],
    )

    indexer_options = engine.available_indexers()
    indexer_key = st.selectbox(
        "Indexer (retrieval strategy)",
        options=[k for k, _ in indexer_options],
        index=_default_index(indexer_options, _initial_for("ub_indexer", settings.default_indexer)),
        format_func=lambda k: dict(indexer_options)[k],
        help=(
            "`semantic` = vector similarity, `syntactic` = BM25 keyword, "
            "`hybrid` = both fused via RRF."
        ),
    )

    rewriter_options = engine.available_rewriters()
    rewriter_choices = [("none", "None")] + rewriter_options
    rewriter_key = st.selectbox(
        "Query rewriter (optional)",
        options=[k for k, _ in rewriter_choices],
        index=_default_index(rewriter_choices, _initial_for("ub_rewriter", settings.default_rewriter)),
        format_func=lambda k: dict(rewriter_choices)[k],
        help=(
            "Transforms the query before retrieval. "
            "`hyde` generates a fake answer to embed; "
            "`multi_query` paraphrases 3 ways."
        ),
    )

    reranker_options = engine.available_rerankers()
    reranker_choices = [("none", "None")] + reranker_options
    reranker_key = st.selectbox(
        "Reranker (optional)",
        options=[k for k, _ in reranker_choices],
        index=_default_index(reranker_choices, _initial_for("ub_reranker", settings.default_reranker)),
        format_func=lambda k: dict(reranker_choices)[k],
        help=(
            "Reorders the top retrieval results with a small CPU model. "
            "First use downloads ~4 MB."
        ),
    )

    st.divider()
    st.subheader(f"Add knowledge to '{current}'")

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
        extensions = engine.connector_extensions(connector_key) or ["txt"]
        uploaded = st.file_uploader(
            f"Upload file ({connector_name})",
            type=extensions,
        )
        if uploaded is not None and st.button(
            f"Index {uploaded.name}", use_container_width=True
        ):
            progress = st.empty()

            def _file_progress(count: int, source: str) -> None:
                progress.caption(
                    f"Processed {count} item{'s' if count != 1 else ''}…"
                )

            with st.spinner("Ingesting…"):
                try:
                    n = engine.ingest(
                        connector_key,
                        indexer_key,
                        uploaded.getvalue(),
                        uploaded.name,
                        on_progress=_file_progress,
                    )
                except Exception as exc:
                    st.error(f"Ingest failed — {_friendly_error(exc)}")
                else:
                    progress.empty()
                    st.success(f"Indexed {n} chunks into '{indexer_key}'")

    elif connector_kind == "url":
        url = st.text_input("URL", placeholder="https://...")
        if url and st.button(
            f"Run {connector_name}", use_container_width=True
        ):
            progress = st.empty()

            def _url_progress(count: int, source: str) -> None:
                progress.caption(
                    f"Fetched {count} page{'s' if count != 1 else ''} so far · "
                    f"latest: {_short(source, 60)}"
                )

            with st.spinner(
                f"{connector_name} running… (crawls may take a few minutes)"
            ):
                try:
                    n = engine.ingest(
                        connector_key,
                        indexer_key,
                        url,
                        url,
                        on_progress=_url_progress,
                    )
                except Exception as exc:
                    st.error(f"Ingest failed — {_friendly_error(exc)}")
                else:
                    progress.empty()
                    if n == 0:
                        st.warning("No content extracted.")
                    else:
                        st.success(f"Indexed {n} chunks into '{indexer_key}'")
    else:
        st.info(
            f"No UI yet for input_kind={connector_kind!r}. Drop a matching "
            f"file under app/connectors/ to extend."
        )

    # ---------- Indexed sources ----------
    st.divider()
    all_sources = engine.list_all_sources()
    with st.expander(
        f"Indexed sources in '{current}' ({len(all_sources)})",
        expanded=False,
    ):
        if not all_sources:
            st.caption("Nothing indexed in this project yet.")
        else:
            for entry in all_sources:
                src_name = entry["source"]
                indexers_label = ", ".join(entry["indexers"])
                total = entry["total"]
                cols = st.columns([5, 1])
                with cols[0]:
                    st.caption(
                        f"**{_short(src_name)}**  \n"
                        f"{total} chunks · {indexers_label}"
                    )
                with cols[1]:
                    if st.button(
                        "Remove",
                        key=f"del::{current}::{src_name}",
                        use_container_width=True,
                    ):
                        n = engine.delete_source(src_name)
                        st.toast(f"Removed {n} chunks from '{src_name}'")
                        st.rerun()

    # ---------- Corpus advisor ----------
    st.divider()
    with st.expander("Corpus advisor", expanded=False):
        st.caption(
            "Samples the current project and asks the LLM to recommend a "
            "retrieval pipeline. Costs one LLM call."
        )
        user_goals = st.text_area(
            "What kinds of questions will users ask?",
            placeholder=(
                "e.g. 'Look up API error codes and rate limits' or "
                "'Summarize and explain key concepts'"
            ),
            key="advisor_goals",
            height=80,
        )
        if st.button("Recommend settings", use_container_width=True):
            with st.spinner("Sampling + asking the LLM…"):
                try:
                    result = engine.recommend_settings(user_goals, provider_key)
                except Exception as exc:
                    result = {"error": _friendly_error(exc)}
            st.session_state["advisor_result"] = result

        result = st.session_state.get("advisor_result")
        if result:
            if "error" in result:
                st.warning(result["error"])
                if "raw" in result:
                    with st.expander("Raw LLM output"):
                        st.code(result["raw"])
            else:
                rec = result["recommendation"]
                st.markdown(
                    f"**Indexer:** `{rec['indexer']}`  \n"
                    f"**Rewriter:** `{rec['rewriter']}`  \n"
                    f"**Reranker:** `{rec['reranker']}`  \n"
                    f"**Chunk size / overlap:** `{rec['chunk_size']}` / `{rec['chunk_overlap']}`"
                )
                st.caption(f"_{rec['rationale']}_")
                if st.button(
                    "Apply (dropdowns only — does not re-index)",
                    use_container_width=True,
                ):
                    st.session_state["ub_indexer"] = rec["indexer"]
                    st.session_state["ub_rewriter"] = rec["rewriter"]
                    st.session_state["ub_reranker"] = rec["reranker"]
                    st.toast("Applied. Re-index sources to use the new chunk sizes.")
                    st.rerun()

    # ---------- Evaluation ----------
    st.divider()
    with st.expander("Evaluation", expanded=False):
        from eval_harness import (
            all_common_configs,
            delete_question,
            load_golden,
            run_eval,
        )

        golden = load_golden(current)
        st.caption(
            f"{len(golden)} question(s) saved in the eval set for '{current}'. "
            "Add more from the 'Save this Q to eval set' button under any "
            "assistant answer in the chat."
        )

        if golden:
            with st.expander(f"View / remove eval questions ({len(golden)})"):
                for q in golden:
                    cols = st.columns([5, 1])
                    with cols[0]:
                        st.caption(f"**Q:** {_short(q.question, 80)}")
                        if q.expected_sources:
                            st.caption(
                                "_Expected: "
                                + ", ".join(_short(s, 30) for s in q.expected_sources)
                                + "_"
                            )
                    with cols[1]:
                        if st.button(
                            "Remove",
                            key=f"eval_del::{current}::{q.question}",
                            use_container_width=True,
                        ):
                            delete_question(current, q.question)
                            st.rerun()

        mode = st.radio(
            "What to test",
            ["Current settings only", "All combinations"],
            horizontal=True,
        )

        if st.button(
            "Run eval",
            use_container_width=True,
            disabled=not golden,
        ):
            indexer_keys = [k for k, _ in engine.available_indexers()]
            rewriter_keys = [k for k, _ in engine.available_rewriters()]
            reranker_keys = [k for k, _ in engine.available_rerankers()]

            if mode == "Current settings only":
                configs = [
                    {
                        "indexer": indexer_key,
                        "rewriter": rewriter_key,
                        "reranker": reranker_key,
                    }
                ]
            else:
                configs = all_common_configs(
                    indexer_keys, rewriter_keys, reranker_keys
                )

            with st.spinner(
                f"Running {len(configs)} configs × {len(golden)} questions…"
            ):
                try:
                    st.session_state["eval_result"] = run_eval(
                        engine, current, configs, provider_key
                    )
                except Exception as exc:
                    st.session_state["eval_result"] = {"error": _friendly_error(exc)}

        eval_result = st.session_state.get("eval_result")
        if eval_result:
            if "error" in eval_result:
                st.warning(eval_result["error"])
            else:
                st.markdown(
                    f"**{eval_result['n_configs']}** configs × "
                    f"**{eval_result['n_questions']}** questions"
                )
                rows = eval_result["summary"]
                lines = [
                    "| Config (indexer / rewriter / reranker) | Avg source recall | N |",
                    "|---|---:|---:|",
                ]
                for row in rows:
                    lines.append(
                        f"| `{row['config']}` | {row['avg_source_recall']:.3f} | {row['n_questions']} |"
                    )
                st.markdown("\n".join(lines))
                st.caption(
                    "Source recall = fraction of expected sources that "
                    "showed up in retrieval. Higher is better."
                )

    st.divider()
    if st.button(f"Reset all indexes in '{current}'", use_container_width=True):
        engine.reset_all()
        st.session_state.pop("messages", None)
        st.session_state.pop("advisor_result", None)
        st.session_state.pop("eval_result", None)
        st.success(f"All indexes in '{current}' cleared")
        st.rerun()

    if st.button("Clear chat only", use_container_width=True):
        st.session_state.pop("messages", None)
        st.toast("Chat cleared")


# ---------- Main: chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])
            _render_save_eval_button(idx, msg, current)

if prompt := st.chat_input(f"Ask a question about '{current}'..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(
            "Rewriting query…" if rewriter_key != "none" else "Searching…"
        ):
            try:
                sources = engine.retrieve(
                    indexer_key,
                    prompt,
                    reranker_key=reranker_key,
                    rewriter_key=rewriter_key,
                    provider_key=provider_key,
                )
            except Exception as exc:
                st.error(f"Retrieval failed — {_friendly_error(exc)}")
                sources = []

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
            full = f"_LLM error — {_friendly_error(exc)}_"
        placeholder.markdown(full)
        if sources:
            _render_sources(sources)
            # Inline save button so it sits right below the answer's sources.
            _render_save_eval_button(
                len(st.session_state.messages),  # this message hasn't been appended yet
                {"role": "assistant", "content": full, "sources": sources},
                current,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": full, "sources": sources}
    )
