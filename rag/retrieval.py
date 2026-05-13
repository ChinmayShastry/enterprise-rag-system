"""
rag/retrieval.py
Hybrid search (ChromaDB vector search + BM25 keyword search),
query rewriting, and cross-encoder reranking.

Key fixes over the original:
- CrossEncoder is loaded here and returned — it was imported but never wired in.
- max_results from RBAC is honoured at every stage, not just at the top level.
- All model/config values come from config.yaml, not hardcoded strings.
"""

import os
import yaml
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────────────────────────
# Resource loading (cached for the session)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models and index…")
def load_resources(api_key: str):
    """
    Load and cache all heavy resources:
      - OpenAI client
      - ChromaDB vectorstore
      - BM25 index (rebuilt from vectorstore documents)
      - CrossEncoder reranker

    Cached by api_key so re-authentication reloads cleanly.
    """
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rag_cfg = cfg["rag"]
    chroma_path = os.getenv("CHROMA_PATH", rag_cfg.get("chroma_path", "./chroma_db"))

    client = OpenAI(api_key=api_key)

    embedding_model = OpenAIEmbeddings(
        model=rag_cfg["embedding_model"],
        openai_api_key=api_key,
    )
    vectorstore = Chroma(
        collection_name=rag_cfg["collection_name"],
        embedding_function=embedding_model,
        persist_directory=chroma_path,
    )

    # Rebuild BM25 index from whatever is in the vectorstore
    all_docs = vectorstore.get()
    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[]])

    # Load the cross-encoder reranker (was missing in original app.py)
    reranker = CrossEncoder(rag_cfg["reranker_model"])

    return client, vectorstore, chunks, bm25, reranker


# ─────────────────────────────────────────────────────────────────
# Query rewriting
# ─────────────────────────────────────────────────────────────────

def rewrite_query(client: OpenAI, question: str, llm_model: str) -> list[str]:
    """
    Ask the LLM to generate 3 alternative phrasings of the user's question.
    This bridges the vocabulary gap between how users phrase questions
    and how documents are written.
    """
    prompt = (
        "Rewrite the question below into 3 different search queries "
        "using alternative words and phrasings.\n"
        "Return ONLY a numbered list. Nothing else.\n\n"
        f"Question: {question}\n"
        "Rewritten queries:"
    )
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    queries = [
        line.split(". ", 1)[1].strip()
        for line in raw.split("\n")
        if line.strip() and line[0].isdigit()
    ]
    return queries


# ─────────────────────────────────────────────────────────────────
# Hybrid search
# ─────────────────────────────────────────────────────────────────

def hybrid_search(
    client: OpenAI,
    vectorstore: Chroma,
    chunks: list[Document],
    bm25: BM25Okapi,
    question: str,
    llm_model: str,
    k: int = 5,
) -> list[Document]:
    """
    Run all query variations (original + 3 rewrites) through both:
      - ChromaDB vector search (semantic similarity)
      - BM25 keyword search (exact term matching)

    Results are deduplicated. The union of both methods gives
    much better recall than either alone.
    """
    queries = rewrite_query(client, question, llm_model)
    queries.append(question)  # Always include the original

    seen: dict[str, Document] = {}

    for query in queries:
        # Vector search
        for doc in vectorstore.similarity_search(query, k=k):
            key = doc.page_content[:120]
            if key not in seen:
                seen[key] = doc

        # BM25 keyword search
        if chunks:
            scores = bm25.get_scores(query.lower().split())
            for idx in scores.argsort()[::-1][:k]:
                if scores[idx] > 0:
                    key = chunks[idx].page_content[:120]
                    if key not in seen:
                        seen[key] = chunks[idx]

    return list(seen.values())


# ─────────────────────────────────────────────────────────────────
# Reranking
# ─────────────────────────────────────────────────────────────────

def rerank(
    reranker: CrossEncoder,
    question: str,
    docs: list[Document],
    top_n: int = 3,
) -> list[Document]:
    """
    Score every retrieved chunk against the question using a cross-encoder.
    Returns the top_n most relevant chunks in ranked order.

    Cross-encoders are much more accurate than cosine similarity for
    relevance scoring because they see the query and document together.
    """
    if not docs:
        return docs

    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
