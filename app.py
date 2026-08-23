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

from rag.access import AccessPolicy
from rag.auth import User, authenticate, authorize_query, authorize_tenant
from rag.documents import delete_document
from rag.generation import get_sources, stream_answer
from rag.ingestion import IngestionError, ingest_upload, roles_cleared_for
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
def load_retriever(api_key: str, tenant_id: str) -> Retriever:
    """
    Cached per (api_key, tenant). The tenant MUST be part of the cache key —
    keying on the API key alone would hand one tenant's retriever, and so one
    tenant's documents, to any other tenant sharing that key.
    """
    return build_retriever(load_settings_cached(), api_key, tenant_id)


@st.cache_resource
def load_query_log(tenant_id: str) -> QueryLog:
    """One audit trail per tenant, so an admin dashboard shows only its own."""
    return QueryLog.for_tenant(tenant_id)


settings = load_settings_cached()
app_cfg = settings.app
rag_cfg = settings.rag

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

def _configured_api_key() -> str | None:
    """
    The deployment's own key, if it has one.

    Streamlit Community Cloud supplies secrets through st.secrets rather than
    the process environment, so checking os.environ alone would prompt every
    visitor of a deployed app for a key that the operator had already set.
    Accessing st.secrets raises when no secrets file exists, which is the
    normal case locally.
    """
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


# Pre-fill API key from the deployment's configuration if it has one
_env_key = _configured_api_key()
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
| Username | Password | Tenant | Role    | Clearance |
|----------|----------|--------|---------|-----------|
| alice    | alice123 | acme   | Admin   | public, internal, confidential |
| bob      | bob123   | acme   | Support | public, internal |
| guest    | guest123 | acme   | Viewer  | public |
| carol    | carol123 | globex | Admin   | public, internal, confidential |
        """)
        st.caption(
            "Clearance decides which documents a role can retrieve. Tenant decides "
            "which corpus exists at all — Carol is an admin, but cannot reach a "
            "single Acme document."
        )

    st.stop()


# ─────────────────────────────────────────────────────────────────
# MAIN APP — user is logged in
# ─────────────────────────────────────────────────────────────────

user: User = st.session_state.get("user")
if not user:
    st.rerun()

permissions = settings.permissions_for(user.role)

# A user whose tenant is missing or misspelt has no safe corpus to fall back
# on, so refuse rather than defaulting to one.
if not authorize_tenant(user, settings):
    st.error(
        f"⛔ Account `{user.username}` is assigned to tenant `{user.tenant or '(none)'}`, "
        "which is not configured. Contact your administrator."
    )
    st.stop()

tenant = settings.tenant(user.tenant)
policy = AccessPolicy.for_user(user, settings)
query_log = load_query_log(tenant.tenant_id)

# config.yaml has always defined can_query per role — now it is enforced.
if not authorize_query(user, settings):
    st.error(f"⛔ The `{user.role}` role is not permitted to query this system.")
    st.stop()

# Load all ML resources (cached per api key AND tenant)
try:
    retriever = load_retriever(st.session_state.api_key, tenant.tenant_id)
except Exception as e:
    st.error(
        f"⚠️ Failed to load resources: {e}\n\n"
        "Make sure you have ingested a document first:\n"
        "```\npython scripts/ingest.py --pdf data/your_document.pdf\n```"
    )
    st.stop()

# Unlabelled chunks are retrievable by nobody, so say so loudly rather than
# letting them look like a retrieval failure.
if retriever.unlabelled_count:
    st.warning(
        f"⚠️ {retriever.unlabelled_count} of {retriever.chunk_count} indexed chunks "
        "have no classification label and are invisible to every role. "
        "Re-upload them to apply a label."
    )

# NOTE: the "no documents indexed" guard deliberately does NOT live here.
# Stopping the script before the sidebar renders would lock an admin out of the
# Documents page — the one place they can fix an empty tenant. The check is
# applied per page, after navigation, further down.

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{app_cfg.icon} {app_cfg.title}")
    st.divider()
    st.markdown(f"**{user.display_name}**")
    st.caption(f"Tenant: `{tenant.tenant_id}` — {tenant.display_name}")
    st.caption(f"Role: `{user.role}`")
    st.caption(f"Clearance: {', '.join(f'`{c}`' for c in sorted(policy.clearance))}")
    st.caption(f"Retrieval depth: {permissions.top_n_rerank} chunks")
    st.divider()

    # Document management is an admin capability: uploading changes what every
    # other user of the tenant can retrieve.
    _pages = ["💬 Chat", "📊 Dashboard"]
    if user.role == "admin":
        _pages.insert(1, "📁 Documents")
    page = st.radio("Navigate", _pages, label_visibility="collapsed")
    st.divider()

    if st.button("Sign Out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.caption(
        f"📦 {retriever.chunk_count} chunks indexed · "
        f"Model: `{rag_cfg.llm_model}`"
    )
    if not retriever.reranking_enabled:
        # Say so rather than silently serving weaker context.
        st.caption(
            "⚠️ Cross-encoder unavailable — results are ordered by hybrid "
            "search alone."
        )


# An empty tenant blocks querying but not document management — an admin has to
# be able to reach the upload page in order to fix it.
if retriever.chunk_count == 0 and page != "📁 Documents":
    if user.role == "admin":
        st.warning(
            f"No documents are indexed for **{tenant.display_name}** yet. "
            "Open **📁 Documents** in the sidebar to upload one."
        )
    else:
        st.warning(
            f"No documents are indexed for **{tenant.display_name}** yet. "
            "An administrator needs to upload one before you can ask questions."
        )
    st.stop()


# ─────────────────────────────────────────────────────────────────
# DOCUMENTS PAGE (admin only — hard gate)
# ─────────────────────────────────────────────────────────────────

if page == "📁 Documents":
    if user.role != "admin":
        st.error("⛔ Access denied. Document management is admin-only.")
        st.stop()

    st.title("📁 Documents")
    st.caption(
        f"Uploading here writes into **{tenant.display_name}**'s own collection "
        f"(`{tenant.collection_name}`). No other tenant can retrieve it."
    )

    upload_tab, manage_tab = st.tabs(["⬆️ Upload", "🗂️ Manage"])

    # ── Upload ───────────────────────────────────────────────────
    with upload_tab:
        uploaded = st.file_uploader(
            "PDF to ingest",
            type=["pdf"],
            help="Scanned PDFs are detected automatically and run through OCR.",
        )

        col_left, col_right = st.columns(2)
        with col_left:
            classification = st.selectbox(
                "Classification",
                options=list(settings.classifications),
                index=list(settings.classifications).index(
                    rag_cfg.default_classification
                ),
                help="Decides which roles can retrieve this document.",
            )
        with col_right:
            doc_title = st.text_input(
                "Title (optional)",
                placeholder=uploaded.name if uploaded else "Defaults to the filename",
            )

        # Show the access consequence before spending anything on embeddings,
        # so "confidential" is not discovered later as an apparent bug.
        cleared = roles_cleared_for(settings, classification)
        if cleared:
            st.info(
                f"Retrievable by: **{', '.join(cleared)}**"
                + (
                    ""
                    if len(cleared) == len(settings.roles)
                    else f" — hidden from: {', '.join(sorted(set(settings.roles) - set(cleared)))}"
                )
            )
        else:
            st.warning(
                f"No role is cleared for `{classification}`, so nobody would be "
                "able to retrieve this document."
            )

        with st.expander("Advanced"):
            force_ocr = st.checkbox(
                "Force OCR",
                help="Use for image-only PDFs that extract as blank pages.",
            )
            custom_doc_id = st.text_input(
                "Document ID (optional)",
                placeholder="Defaults to a slug of the filename",
                help="Re-uploading with the same ID replaces that document "
                     "rather than duplicating it.",
            )

        if st.button(
            "Ingest document",
            type="primary",
            disabled=uploaded is None,
            use_container_width=True,
        ):
            status = st.status("Ingesting…", expanded=True)
            try:
                result = ingest_upload(
                    uploaded.getvalue(),
                    uploaded.name,
                    settings=settings,
                    tenant_id=tenant.tenant_id,
                    api_key=st.session_state.api_key,
                    doc_id=custom_doc_id or None,
                    title=doc_title or None,
                    classification=classification,
                    force_ocr=force_ocr,
                    on_progress=lambda message: status.write(message),
                )
            except IngestionError as e:
                status.update(label="Ingestion failed", state="error")
                st.error(f"⚠️ {e}")
            except Exception as e:  # noqa: BLE001 - surface anything to the operator
                status.update(label="Ingestion failed", state="error")
                st.error(f"⚠️ Unexpected error: {e}")
            else:
                status.update(label="Ingestion complete", state="complete")
                st.success(
                    f"Stored **{result.title}** as `{result.doc_id}` — "
                    f"{result.chunk_count} chunks from {result.page_count} pages"
                    + (" (via OCR)" if result.used_ocr else "")
                    + "."
                )
                if result.replaced:
                    st.caption(
                        f"♻️ Replaced {result.replaced_chunks} chunks from the "
                        "previous version of this document."
                    )

                # The cached retriever still holds the pre-upload chunk list and
                # BM25 index, so without this the new document stays invisible
                # until the process restarts.
                load_retriever.clear()
                st.rerun()

    # ── Manage ───────────────────────────────────────────────────
    with manage_tab:
        docs = retriever.documents()
        if not docs:
            st.caption("Nothing indexed for this tenant yet.")
        else:
            st.table({
                "Document ID": [d.doc_id for d in docs],
                "Title": [d.title for d in docs],
                "Classification": [d.classification for d in docs],
                "Chunks": [d.chunk_count for d in docs],
            })

            st.subheader("Delete a document")
            target = st.selectbox(
                "Document",
                options=[d.doc_id for d in docs],
                format_func=lambda doc_id: next(
                    f"{d.title} ({d.doc_id})" for d in docs if d.doc_id == doc_id
                ),
            )
            st.warning(
                "This permanently removes every chunk of that document from "
                f"**{tenant.display_name}**'s collection."
            )
            confirm = st.checkbox(f"Yes, delete `{target}`")
            if st.button(
                "Delete document",
                type="secondary",
                disabled=not confirm,
                use_container_width=True,
            ):
                removed = delete_document(retriever.vectorstore, target)
                load_retriever.clear()
                st.success(f"Deleted `{target}` ({removed} chunks).")
                st.rerun()


# ─────────────────────────────────────────────────────────────────
# CHAT PAGE
# ─────────────────────────────────────────────────────────────────

elif page == "💬 Chat":
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
                            tenant=tenant.tenant_id,
                        )
                        st.session_state.voted.add(_mid)
                        st.rerun()
                with cols[1]:
                    if st.button("👎", key=f"dn_{_mid}"):
                        query_log.save_feedback(
                            user.username, user.role,
                            entry["question"], entry["answer"], "not_useful",
                            tenant=tenant.tenant_id,
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
                    policy=policy,
                )

            if not top_docs:
                st.info(
                    "No documents matching your clearance "
                    f"(`{'`, `'.join(sorted(policy.clearance)) or 'none'}`) "
                    "contain an answer to that question."
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
        query_log.log_query(
            user.username, user.role, question, answer, sources,
            tenant=tenant.tenant_id,
        )
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

    # ── Indexed documents ─────────────────────────────────────────
    st.subheader("Indexed Documents")
    docs = retriever.documents()
    if docs:
        st.table({
            "Document ID": [d.doc_id for d in docs],
            "Title": [d.title for d in docs],
            "Classification": [d.classification for d in docs],
            "Chunks": [d.chunk_count for d in docs],
        })
        st.caption(
            f"Remove one with: `python scripts/documents.py delete "
            f"--tenant {tenant.tenant_id} --doc-id <id>`"
        )
    else:
        st.caption("No documents carry a doc_id — re-ingest to populate this view.")

    st.divider()

    # ── System health ─────────────────────────────────────────────
    st.subheader("System Health")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(f"- **Tenant:** `{tenant.tenant_id}` — {tenant.display_name}")
        st.markdown(f"- **Indexed chunks:** {retriever.chunk_count}")
        st.markdown(f"- **Vector DB:** ChromaDB · `{tenant.collection_name}`")
        st.markdown(f"- **LLM:** `{rag_cfg.llm_model}`")
    with h2:
        st.markdown(f"- **Embeddings:** `{rag_cfg.embedding_model}`")
        st.markdown(f"- **Reranker:** `{rag_cfg.reranker_model}`")
        st.markdown(f"- **BM25 index:** {retriever.chunk_count} chunks")
        st.markdown(f"- **Unlabelled chunks:** {retriever.unlabelled_count}")

    st.subheader("Role Clearances")
    st.table({
        "Role": list(settings.roles),
        "Can query": [p.can_query for p in settings.roles.values()],
        "Clearance": [", ".join(sorted(p.clearance)) or "—" for p in settings.roles.values()],
        "Depth": [p.top_n_rerank for p in settings.roles.values()],
    })
