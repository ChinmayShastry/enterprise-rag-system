import os
import json
import datetime
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
LOG_FILE       = "logs/query_log.jsonl"
FEEDBACK_FILE  = "logs/feedback_log.jsonl"
os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────
# USERS & ROLES
# ─────────────────────────────────────────
USERS = {
    "alice": {"password": "alice123", "role": "admin"},
    "bob":   {"password": "bob123",   "role": "support"},
    "guest": {"password": "guest123", "role": "viewer"},
}
ROLE_PERMISSIONS = {
    "admin":   {"can_query": True, "can_see_sources": True,  "max_results": 8},
    "support": {"can_query": True, "can_see_sources": True,  "max_results": 5},
    "viewer":  {"can_query": True, "can_see_sources": False, "max_results": 3},
}

# ─────────────────────────────────────────
# LOAD MODELS (cached so they load once)
# ─────────────────────────────────────────
@st.cache_resource
def load_resources(api_key):
    client = OpenAI(api_key=api_key)

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    vectorstore = Chroma(
        collection_name="lloyd_manual",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )

    # Rebuild BM25 from vectorstore chunks
    all_docs = vectorstore.get()
    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)


    return client, vectorstore, chunks, bm25

# ─────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────
def rewrite_query(client, question):
    prompt = f"""Rewrite the question below into 3 different search queries using alternative words.
Return ONLY a numbered list. Nothing else.

Question: {question}
Rewritten queries:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()
    queries = [
        line.split(". ", 1)[1].strip()
        for line in raw.split("\n")
        if line.strip() and line[0].isdigit()
    ]
    return queries

def hybrid_search(client, vectorstore, chunks, bm25, question, k=3):
    queries = rewrite_query(client, question)
    queries.append(question)
    seen = {}
    for query in queries:
        for doc in vectorstore.similarity_search(query, k=k):
            key = doc.page_content[:100]
            if key not in seen:
                seen[key] = doc
        scores = bm25.get_scores(query.lower().split())
        for idx in scores.argsort()[::-1][:k]:
            if scores[idx] > 0:
                key = chunks[idx].page_content[:100]
                if key not in seen:
                    seen[key] = chunks[idx]
    return list(seen.values())

def rerank(reranker, question, docs, top_n=3):
    pairs  = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:top_n]]

def generate_answer(client, question, docs):
    context = ""
    sources = []
    for i, doc in enumerate(docs):
        context += f"[Chunk {i+1} - Page {doc.metadata['page']}]\n{doc.page_content}\n\n"
        sources.append(f"Page {doc.metadata['page']}")
    prompt = f"""You are a helpful assistant for Lloyd washing machine users.
Answer using ONLY the context below. If the answer is not there, say so.
Always mention which page your answer comes from.

Context:
{context}

Question: {question}
Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content, list(set(sources))

def log_query(user, question, answer, sources):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": user["username"],
        "role": user["role"],
        "question": question,
        "answer": answer,
        "sources": sources
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def save_feedback(user, question, answer, feedback):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": user["username"],
        "role": user["role"],
        "question": question,
        "answer": answer[:200],
        "feedback": feedback
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────
st.set_page_config(page_title="Lloyd Knowledge AI", page_icon="🤖", layout="wide")

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None

# ── LOGIN SCREEN ──
if not st.session_state.logged_in:
    st.title("🤖 Lloyd Knowledge AI Assistant")
    st.subheader("Please log in to continue")

    col1, col2 = st.columns([1, 2])
    with col1:
        api_key  = st.text_input("🔑 OpenAI API Key", type="password",
                                  placeholder="sk-...")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if not api_key or not api_key.startswith("sk-"):
                st.error("❌ Please enter a valid OpenAI API key (starts with sk-)")
            else:
                user_data = USERS.get(username)
                if user_data and user_data["password"] == password:
                    st.session_state.logged_in  = True
                    st.session_state.api_key    = api_key
                    st.session_state.user       = {
                        "username": username,
                        "role": user_data["role"],
                        "permissions": ROLE_PERMISSIONS[user_data["role"]]
                    }
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

    with col2:
        st.info("**Demo accounts:**\n\n👤 alice / alice123 (admin)\n\n👤 bob / bob123 (support)\n\n👤 guest / guest123 (viewer)")
        st.warning("⚠️ Your OpenAI API key is used only for your session and is never stored.")

# ── MAIN APP ──
else:
    user        = st.session_state.user
    permissions = user["permissions"]

    # Load resources
    client, vectorstore, chunks, bm25 = load_resources(st.session_state.api_key)

    # Sidebar
    with st.sidebar:
        st.title("🤖 Lloyd AI")
        st.markdown(f"**User:** {user['username']}")
        st.markdown(f"**Role:** `{user['role']}`")
        st.divider()
        page = st.radio("Navigate", ["💬 Chat", "📊 Dashboard"])
        st.divider()
        if st.button("Logout"):
            st.session_state.logged_in  = False
            st.session_state.user       = None
            st.session_state.chat_history = []
            st.rerun()

    # ── CHAT PAGE ──
    if page == "💬 Chat":
        st.title("💬 Lloyd Washing Machine Assistant")

        # Show chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                if permissions["can_see_sources"] and chat["sources"]:
                    st.caption(f"📄 Sources: {', '.join(chat['sources'])}")

        # Chat input
        question = st.chat_input("Ask anything about your Lloyd washing machine...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching manual..."):
                    # Full pipeline
                    # raw_docs   = hybrid_search(client, vectorstore, chunks, bm25, question, k=8)
                    # top_docs   = rerank(reranker, question, raw_docs, top_n=5)
                    # raw_docs   = hybrid_search(client, vectorstore, chunks, bm25, question, k=8)
                    # top_docs   = raw_docs[:5]  # Skip re-ranker temporarily
                    top_docs   = hybrid_search(client, vectorstore, chunks, bm25, question, k=5)
                    answer, sources = generate_answer(client, question, top_docs)

                st.write(answer)
                if permissions["can_see_sources"]:
                    st.caption(f"📄 Sources: {', '.join(sources)}")

                # Feedback buttons
                st.markdown("**Was this helpful?**")
                col1, col2 = st.columns([1, 8])
                with col1:
                    if st.button("👍", key=f"up_{len(st.session_state.chat_history)}"):
                        save_feedback(user, question, answer, "useful")
                        st.success("Thanks!")
                with col2:
                    if st.button("👎", key=f"down_{len(st.session_state.chat_history)}"):
                        save_feedback(user, question, answer, "not_useful")
                        st.success("Thanks for the feedback!")

            # Log and save to history
            log_query(user, question, answer, sources)
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

    # ── DASHBOARD PAGE ──
    elif page == "📊 Dashboard":
        st.title("📊 Evaluation Dashboard")

        if user["role"] != "admin":
            st.warning("⚠️ Dashboard is only available to admins.")
        else:
            # Query stats
            queries = []
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    queries = [json.loads(l) for l in f.readlines()]

            feedbacks = []
            if os.path.exists(FEEDBACK_FILE):
                with open(FEEDBACK_FILE, "r") as f:
                    feedbacks = [json.loads(l) for l in f.readlines()]

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            useful     = sum(1 for fb in feedbacks if fb["feedback"] == "useful")
            not_useful = sum(1 for fb in feedbacks if fb["feedback"] == "not_useful")
            score      = (useful / len(feedbacks) * 100) if feedbacks else 0

            col1.metric("Total Queries",    len(queries))
            col2.metric("Total Feedback",   len(feedbacks))
            col3.metric("👍 Useful",        useful)
            col4.metric("🎯 Satisfaction",  f"{score:.1f}%")

            st.divider()

            # Recent queries table
            st.subheader("📋 Recent Queries")
            if queries:
                for q in reversed(queries[-10:]):
                    with st.expander(f"[{q['username']}] {q['question'][:60]}..."):
                        st.markdown(f"**Time:** {q['timestamp']}")
                        st.markdown(f"**Role:** {q['role']}")
                        st.markdown(f"**Answer:** {q['answer'][:300]}...")
                        st.markdown(f"**Sources:** {', '.join(q['sources'])}")

            st.divider()

            # System health
            st.subheader("⚙️ System Health")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"- **Vector DB chunks:** {vectorstore._collection.count()}")
                st.markdown(f"- **BM25 index size:** {len(chunks)} chunks")
                st.markdown(f"- **LLM:** gpt-4o-mini")
            with col2:
                st.markdown(f"- **Embeddings:** text-embedding-3-small")
                st.markdown(f"- **Re-ranker:** ms-marco-MiniLM-L-6-v2")
                st.markdown(f"- **Log file:** {LOG_FILE}")