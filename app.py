"""
app.py — Enterprise RAG Assistant
Streamlit frontend for the RAG system.

This file is the only Streamlit-aware layer. Everything under rag/ is plain
Python, so the same retrieval and generation code runs from the CLI
(scripts/query.py), from tests, and from any future API server.

Caching lives here because caching is a UI concern: @st.cache_resource wraps
the framework-free factories rather than being baked into them.
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from rag.auth import User, authenticate, authorize_query
from rag.generation import get_sources, stream_answer
from rag.logger import QueryLog
from rag.retrieval import Retriever, build_retriever
from rag.settings import Settings, get_settings

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Cached resource loaders — the Streamlit boundary
# ─────────────────────────────────────────────────────────────────


@st.cache_resource
def load_settings_cached() -> Settings:
    return get_settings()


@st.cache_resource(show_spinner="Loading models and index…")
def load_retriever(api_key: str) -> Retriever:
    return build_retriever(load_settings_cached(), api_key)


@st.cache_resource
def load_query_log() -> QueryLog:
    return QueryLog()


settings = load_settings_cached()
app_cfg = settings.app
rag_cfg = settings.rag
query_log = load_query_log()

st.set_page_config(
    page_title=app_cfg.title,
    page_icon=app_cfg.icon,
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "logged_in": False,
    "user": None,
    "api_key": None,
    "chat_history": [],  # list of {id, question, answer, sources}
    "voted": set(),      # set of chat entry IDs that have received feedback
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Pre-fill API key from environment if available
_env_key = os.getenv("OPENAI_API_KEY")
if _env_key and not st.session_state.api_key:
    st.session_state.api_key = _env_key


# ─────────────────────────────────────────────────────────────────
# LOGIN SCREEN
# ─────────────────────────────────────────────────────────────────

if not st.session_state.logged_in:
    st.title(f"{app_cfg.icon} {app_cfg.title}")
    st.markdown(f"*{app_cfg.description}*")
    st.divider()

    col_form, col_info = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Sign In")

        # Only ask for API key if not in environment
        if not _env_key:
            api_key_input = st.text_input(
                "🔑 OpenAI API Key",
                type="password",
                placeholder="sk-…",
                help="Your key is used only for this session and never stored.",
            )
        else:
            api_key_input = _env_key
            st.info("🔑 API key loaded from environment.", icon="✅")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Sign In", use_container_width=True, type="primary"):
            if not api_key_input or not api_key_input.startswith("sk-"):
                st.error("Please enter a valid OpenAI API key (starts with sk-).")
            else:
                user = authenticate(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.api_key = api_key_input
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with col_info:
        st.subheader("Demo Accounts")
        st.markdown("""
| Username | Password | Role    | Sources | Depth |
|----------|----------|---------|---------|-------|
| alice    | alice123 | Admin   | ✅ Visible | 5 chunks |
| bob      | bob123   | Support | ✅ Visible | 3 chunks |
| guest    | guest123 | Viewer  | ❌ Hidden  | 2 chunks |
        """)
        st.caption(
            "Different roles retrieve different amounts of context and control "
            "what metadata is exposed in the UI."
        )

    st.stop()


# ─────────────────────────────────────────────────────────────────
# MAIN APP — user is logged in
# ─────────────────────────────────────────────────────────────────

user: User = st.session_state.get("user")
if not user:
    st.rerun()

permissions = settings.permissions_for(user.role)

# config.yaml has always defined can_query per role — now it is enforced.
if not authorize_query(user, settings):
    st.error(f"⛔ The `{user.role}` role is not permitted to query this system.")
    st.stop()

# Load all ML resources (cached after first load)
try:
    retriever = load_retriever(st.session_state.api_key)
except Exception as e:
    st.error(
        f"⚠️ Failed to load resources: {e}\n\n"
        "Make sure you have ingested a document first:\n"
        "```\npython scripts/ingest.py --pdf data/your_document.pdf\n```"
    )
    st.stop()

if retriever.chunk_count == 0:
    st.warning(
        "No documents are indexed yet. Run the ingestion script first:\n"
        "```\npython scripts/ingest.py --pdf data/your_document.pdf\n```"
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{app_cfg.icon} {app_cfg.title}")
    st.divider()
    st.markdown(f"**{user.display_name}**")
    st.caption(f"Role: `{user.role}`")
    st.caption(f"Retrieval depth: {permissions.top_n_rerank} chunks")
    st.divider()

    page = st.radio("Navigate", ["💬 Chat", "📊 Dashboard"], label_visibility="collapsed")
    st.divider()

    if st.button("Sign Out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.caption(
        f"📦 {retriever.chunk_count} chunks indexed · "
        f"Model: `{rag_cfg.llm_model}`"
    )


# ─────────────────────────────────────────────────────────────────
# CHAT PAGE
# ─────────────────────────────────────────────────────────────────

if page == "💬 Chat":
    st.title("💬 Chat")

    # Render conversation history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])

        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if permissions.can_see_sources and entry["sources"]:
                st.caption(f"📄 Sources: {', '.join(entry['sources'])}")

            # Feedback — use stored ID to prevent key collisions and double-votes
            _mid = entry["id"]
            if _mid in st.session_state.voted:
                st.caption("✅ Feedback recorded")
            else:
                cols = st.columns([1, 1, 10])
                with cols[0]:
                    if st.button("👍", key=f"up_{_mid}"):
                        query_log.save_feedback(
                            user.username, user.role,
                            entry["question"], entry["answer"], "useful",
                        )
                        st.session_state.voted.add(_mid)
                        st.rerun()
                with cols[1]:
                    if st.button("👎", key=f"dn_{_mid}"):
                        query_log.save_feedback(
                            user.username, user.role,
                            entry["question"], entry["answer"], "not_useful",
                        )
                        st.session_state.voted.add(_mid)
                        st.rerun()

    # Chat input
    question = st.chat_input("Ask anything about your documents…")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            # Retrieval phase — show spinner while searching
            with st.spinner("🔍 Searching documents…"):
                top_docs = retriever.retrieve(
                    question,
                    max_results=permissions.max_results,
                    top_n=permissions.top_n_rerank,
                )

            sources = get_sources(top_docs)

            # Stream the answer token by token
            answer = st.write_stream(
                stream_answer(
                    retriever.client, question, top_docs,
                    llm_model=rag_cfg.llm_model,
                    persona=app_cfg.persona,
                )
            )

            if permissions.can_see_sources and sources:
                st.caption(f"📄 Sources: {', '.join(sources)}")

            new_id = str(uuid.uuid4())[:8]

        # Persist to history and log
        query_log.log_query(user.username, user.role, question, answer, sources)
        st.session_state.chat_history.append({
            "id": new_id,
            "question": question,
            "answer": answer,
            "sources": sources,
        })
        # Re-render so the new turn picks up its feedback buttons from history.
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# DASHBOARD PAGE (admin only — hard gate)
# ─────────────────────────────────────────────────────────────────

elif page == "📊 Dashboard":
    if user.role != "admin":
        st.error("⛔ Access denied. The dashboard is only available to admin users.")
        st.stop()

    st.title("📊 Evaluation Dashboard")

    queries = query_log.load_queries()
    feedbacks = query_log.load_feedback()

    # ── Top metrics ──────────────────────────────────────────────
    useful = sum(1 for f in feedbacks if f["feedback"] == "useful")
    not_useful = sum(1 for f in feedbacks if f["feedback"] == "not_useful")
    satisfaction = round(useful / len(feedbacks) * 100, 1) if feedbacks else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Queries", len(queries))
    m2.metric("Feedback Received", len(feedbacks))
    m3.metric("👍 Useful", useful)
    m4.metric("🎯 Satisfaction", f"{satisfaction}%")

    st.divider()

    # ── Queries by role ───────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Queries by Role")
        if queries:
            role_counts: dict[str, int] = {}
            for q in queries:
                role_counts[q["role"]] = role_counts.get(q["role"], 0) + 1
            st.bar_chart(role_counts)
        else:
            st.caption("No queries logged yet.")

    with col_right:
        st.subheader("Feedback Breakdown")
        if feedbacks:
            st.bar_chart({"👍 Useful": useful, "👎 Not Useful": not_useful})
        else:
            st.caption("No feedback logged yet.")

    st.divider()

    # ── Recent queries ────────────────────────────────────────────
    st.subheader("Recent Queries")
    if queries:
        for entry in reversed(queries[-15:]):
            label = f"[{entry['username']} · {entry['role']}]  {entry['question'][:70]}"
            with st.expander(label):
                st.caption(f"🕐 {entry['timestamp']}")
                st.markdown(f"**Answer:** {entry['answer'][:400]}…")
                if entry.get("sources"):
                    st.caption(f"📄 Sources: {', '.join(entry['sources'])}")
    else:
        st.caption("No queries logged yet.")

    st.divider()

    # ── System health ─────────────────────────────────────────────
    st.subheader("System Health")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(f"- **Indexed chunks:** {retriever.chunk_count}")
        st.markdown(f"- **Vector DB:** ChromaDB · `{rag_cfg.collection_name}`")
        st.markdown(f"- **LLM:** `{rag_cfg.llm_model}`")
    with h2:
        st.markdown(f"- **Embeddings:** `{rag_cfg.embedding_model}`")
        st.markdown(f"- **Reranker:** `{rag_cfg.reranker_model}`")
        st.markdown(f"- **BM25 index:** {retriever.chunk_count} chunks")
