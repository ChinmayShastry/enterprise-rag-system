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

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from rag.access import AccessPolicy, unlabelled
from rag.settings import Settings, TenantConfig


class TenantMismatchError(RuntimeError):
    """
    Raised when a retriever is handed a policy belonging to another tenant.

    This should be unreachable, and that is the point: the most likely way it
    fires is a cached retriever being reused across tenants, which would
    otherwise be a silent cross-tenant read.
    """

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


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
        bm25: "BM25Okapi | None",
        reranker: "CrossEncoder | None",
        settings: Settings,
        tenant: TenantConfig,
    ):
        self.client = client
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.bm25 = bm25
        self.reranker = reranker
        self.settings = settings
        self.tenant = tenant

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def documents(self) -> list:
        """Inventory of the documents in this tenant's collection."""
        from rag.documents import list_documents

        return list_documents(self.vectorstore)

    @property
    def unlabelled_count(self) -> int:
        """
        Chunks with no classification label. They are invisible to every role,
        so a non-zero count here means part of the index was ingested before
        labelling and needs re-ingesting.
        """
        return len(unlabelled(self.chunks))

    def _assert_same_tenant(self, policy: AccessPolicy) -> None:
        """
        Refuse to serve a policy from another tenant.

        Collection-per-tenant already makes cross-tenant retrieval impossible
        in practice — this retriever only ever holds one tenant's collection.
        The check exists because the realistic failure is not a bad filter but
        a wrongly cached retriever, and an exception is far better than
        quietly answering one customer from another's documents.
        """
        if policy.tenant != self.tenant.tenant_id:
            raise TenantMismatchError(
                f"Retriever is bound to tenant '{self.tenant.tenant_id}' but was "
                f"given a policy for tenant '{policy.tenant}'."
            )

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

    def search(self, question: str, k: int, *, policy: AccessPolicy) -> list[Document]:
        """
        Run all query variations (original + 3 rewrites) through both
        vector search and BM25 keyword search, then deduplicate.
        The union of both methods gives better recall than either alone.

        `policy` is required, not optional — an unfiltered search is never the
        right default, and a keyword argument makes every call site declare
        whose access it is running under.
        """
        self._assert_same_tenant(policy)

        if policy.denies_everything:
            return []

        # An empty collection cannot match anything, and rewrite_query() costs
        # an LLM round trip. Bail before spending it — this is the normal state
        # for a tenant that has been created but not yet had a document
        # ingested.
        if not self.chunks:
            return []

        where = policy.where_clause()
        queries = self.rewrite_query(question)
        queries.append(question)  # Always include the original

        seen: dict[str, Document] = {}

        for query in queries:
            # Filtered inside the vector store, so denied chunks never leave it.
            for doc in self.vectorstore.similarity_search(query, k=k, filter=where):
                seen.setdefault(doc.page_content[:120], doc)

            # BM25 searches an in-memory list that Chroma's filter cannot reach,
            # so clearance is applied here while picking the top k. Denied chunks
            # are skipped rather than consuming a slot, so a viewer still gets k
            # results from the documents they are allowed to see.
            #
            # bm25 is None when the collection has no indexed terms; the vector
            # path above still runs, it just returns nothing.
            if self.bm25 is not None and self.chunks:
                scores = self.bm25.get_scores(query.lower().split())
                taken = 0
                for idx in scores.argsort()[::-1]:
                    if taken >= k or scores[idx] <= 0:
                        break  # scores descend; nothing better remains
                    chunk = self.chunks[idx]
                    if not policy.permits(chunk):
                        continue
                    seen.setdefault(chunk.page_content[:120], chunk)
                    taken += 1

        return list(seen.values())

    # ── reranking ────────────────────────────────────────────────

    @property
    def reranking_enabled(self) -> bool:
        """
        False when the cross-encoder could not be loaded. Answers are still
        produced, from the hybrid search order rather than a relevance score,
        so the UI should say so rather than silently serving weaker context.
        """
        return self.reranker is not None

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

        Falls back to hybrid-search order when no cross-encoder is available —
        see build_retriever(). Degraded relevance beats a dead application.
        """
        if not docs:
            return docs

        if self.reranker is None:
            return docs[:top_n]

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
        policy: AccessPolicy,
    ) -> list[Document]:
        """
        Search then rerank — the whole retrieval path in one call.

        The final policy.filter() is deliberate redundancy: both search paths
        already enforce clearance, so this should never drop anything. It is
        the backstop that keeps a future bug in either path from putting an
        unauthorised chunk in front of the LLM.
        """
        # Checked here as well as in search(), so a subclass or future rewrite
        # of search() cannot skip the tenant check.
        self._assert_same_tenant(policy)

        candidates = self.search(question, k=max_results, policy=policy)
        ranked = self.rerank(question, candidates, top_n=top_n)
        return policy.filter(ranked)


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────


def _load_reranker(model_name: str):
    """
    Load the cross-encoder, or return None if it is unavailable.

    sentence-transformers pulls in torch, which is by far the heaviest
    dependency here and the one most likely to be missing or to exhaust memory
    on a constrained host — a free-tier PaaS dyno, for instance. Reranking
    improves relevance but is not required to answer a question, so the whole
    application should not fail to start over it.

    The caller can check Retriever.reranking_enabled and tell the user that
    results are ordered by hybrid search alone.
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:  # ImportError, or a torch that fails to initialise
        logger.warning(
            "Cross-encoder unavailable (%s: %s); falling back to hybrid-search "
            "order. Install sentence-transformers to restore reranking.",
            type(e).__name__,
            e,
        )
        return None

    try:
        return CrossEncoder(model_name)
    except Exception as e:  # download failure, corrupt cache, out of memory
        logger.warning(
            "Could not load reranker model %r (%s: %s); falling back to "
            "hybrid-search order.",
            model_name,
            type(e).__name__,
            e,
        )
        return None


def build_retriever(settings: Settings, api_key: str, tenant_id: str) -> Retriever:
    """
    Load one tenant's vector store, rebuild its BM25 index, and load the
    cross-encoder.

    The retriever is bound to a single tenant's collection, so a caller holding
    it can only ever read that tenant's documents. Callers that cache this MUST
    include tenant_id in the cache key.

    This is expensive — the caller decides how to cache it. app.py uses
    @st.cache_resource; a long-lived server would build one per tenant.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from openai import OpenAI
    from rank_bm25 import BM25Okapi

    rag_cfg = settings.rag
    tenant = settings.tenant(tenant_id)  # raises UnknownTenantError

    client = OpenAI(api_key=api_key)

    embeddings = OpenAIEmbeddings(model=rag_cfg.embedding_model, api_key=api_key)
    vectorstore = Chroma(
        collection_name=tenant.collection_name,
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
    # BM25Okapi cannot be built from a corpus with no terms: computing average
    # idf divides by the vocabulary size, so an empty collection raised
    # ZeroDivisionError before the caller ever got to check chunk_count. A
    # tenant with nothing ingested yet is a normal state — a newly onboarded
    # customer — so leave the index unset and let search() skip the keyword
    # path. `any()` rather than a plain truthiness check also covers a corpus
    # of chunks that are all whitespace.
    bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    reranker = _load_reranker(rag_cfg.reranker_model)

    return Retriever(
        client=client,
        vectorstore=vectorstore,
        chunks=chunks,
        bm25=bm25,
        reranker=reranker,
        settings=settings,
        tenant=tenant,
    )
