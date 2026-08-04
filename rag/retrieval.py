"""
rag/retrieval.py
Hybrid search (ChromaDB vector search + BM25 keyword search),
query rewriting, and cross-encoder reranking.

This module contains no Streamlit code. It previously used
@st.cache_resource, which meant retrieval could only run inside a Streamlit
session — no API server, no batch job, no test. Caching is now the caller's
concern: app.py wraps build_retriever() in @st.cache_resource, while the CLI
and tests call it directly.

Heavy dependencies (torch via sentence-transformers, chromadb) are imported
inside build_retriever() rather than at module scope, so `import
rag.retrieval` stays cheap and Retriever's logic can be unit-tested with
stand-in collaborators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from rag.settings import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────


class Retriever:
    """
    Owns the retrieval pipeline for one document collection.

    Collaborators are injected rather than constructed here, so tests can
    supply fakes for the OpenAI client, vector store, and reranker.
    Use build_retriever() for the real thing.
    """

    def __init__(
        self,
        *,
        client: "OpenAI",
        vectorstore: Any,
        chunks: list[Document],
        bm25: "BM25Okapi",
        reranker: "CrossEncoder",
        settings: Settings,
    ):
        self.client = client
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.bm25 = bm25
        self.reranker = reranker
        self.settings = settings

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    # ── query rewriting ──────────────────────────────────────────

    def rewrite_query(self, question: str) -> list[str]:
        """
        Ask the LLM for 3 alternative phrasings of the question.
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
        response = self.client.chat.completions.create(
            model=self.settings.rag.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        return [
            line.split(". ", 1)[1].strip()
            for line in raw.split("\n")
            if line.strip() and line[0].isdigit()
        ]

    # ── hybrid search ────────────────────────────────────────────

    def search(self, question: str, k: int = 5) -> list[Document]:
        """
        Run all query variations (original + 3 rewrites) through both
        vector search and BM25 keyword search, then deduplicate.
        The union of both methods gives better recall than either alone.
        """
        queries = self.rewrite_query(question)
        queries.append(question)  # Always include the original

        seen: dict[str, Document] = {}

        for query in queries:
            for doc in self.vectorstore.similarity_search(query, k=k):
                seen.setdefault(doc.page_content[:120], doc)

            if self.chunks:
                scores = self.bm25.get_scores(query.lower().split())
                for idx in scores.argsort()[::-1][:k]:
                    if scores[idx] > 0:
                        chunk = self.chunks[idx]
                        seen.setdefault(chunk.page_content[:120], chunk)

        return list(seen.values())

    # ── reranking ────────────────────────────────────────────────

    def rerank(
        self,
        question: str,
        docs: list[Document],
        top_n: int = 3,
    ) -> list[Document]:
        """
        Score every retrieved chunk against the question with a cross-encoder,
        which reads query and document together and is far more accurate than
        cosine similarity alone. Returns the top_n in ranked order.
        """
        if not docs:
            return docs

        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]

    # ── full pipeline ────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        *,
        max_results: int,
        top_n: int,
    ) -> list[Document]:
        """Search then rerank — the whole retrieval path in one call."""
        return self.rerank(question, self.search(question, k=max_results), top_n=top_n)


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────


def build_retriever(settings: Settings, api_key: str) -> Retriever:
    """
    Load the vector store, rebuild the BM25 index, and load the cross-encoder.

    This is expensive — the caller decides how to cache it. app.py uses
    @st.cache_resource; a long-lived server would build it once at startup.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from openai import OpenAI
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    rag_cfg = settings.rag

    client = OpenAI(api_key=api_key)

    embeddings = OpenAIEmbeddings(model=rag_cfg.embedding_model, api_key=api_key)
    vectorstore = Chroma(
        collection_name=rag_cfg.collection_name,
        embedding_function=embeddings,
        persist_directory=str(rag_cfg.chroma_path),
    )

    # Rebuild the BM25 index from whatever is currently in the vector store.
    stored = vectorstore.get()
    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(stored["documents"], stored["metadatas"])
    ]
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[]])

    reranker = CrossEncoder(rag_cfg.reranker_model)

    return Retriever(
        client=client,
        vectorstore=vectorstore,
        chunks=chunks,
        bm25=bm25,
        reranker=reranker,
        settings=settings,
    )
