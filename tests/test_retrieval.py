"""
Retriever pipeline tests using stand-in collaborators.

None of this needs an API key, a running ChromaDB, or torch — which is the
practical payoff of taking Streamlit (and the @st.cache_resource decorator)
out of rag/retrieval.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.documents import Document

from rag.access import AccessPolicy
from rag.retrieval import Retriever


class FakeClient:
    """Mimics the slice of the OpenAI client that rewrite_query touches."""

    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeVectorStore:
    """
    Stands in for Chroma, and honours the metadata filter the way Chroma does.

    Applying the filter here matters: if this fake ignored it, the RBAC tests
    would pass on the post-retrieval backstop alone and a broken where-clause
    would go unnoticed.
    """

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.queries: list[tuple[str, int]] = []
        self.filters: list[dict | None] = []

    def similarity_search(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[Document]:
        self.queries.append((query, k))
        self.filters.append(filter)

        candidates = self.docs
        if filter:
            allowed = filter["classification"]["$in"]
            candidates = [
                d for d in candidates if d.metadata.get("classification") in allowed
            ]
        return candidates[:k]


class FakeBM25:
    def __init__(self, scores: list[float]):
        self._scores = np.array(scores, dtype=float)

    def get_scores(self, tokens):
        return self._scores


class FakeReranker:
    """Scores each pair by how many query words appear in the chunk."""

    def predict(self, pairs):
        return [
            sum(word in chunk.lower() for word in question.lower().split())
            for question, chunk in pairs
        ]


REWRITES = "1. alternative one\n2. alternative two\n3. alternative three"


def make_docs(*texts: str, classification: str = "public") -> list[Document]:
    return [
        Document(
            page_content=t,
            metadata={"page": i + 1, "classification": classification},
        )
        for i, t in enumerate(texts)
    ]


def build(settings, *, docs=None, bm25_scores=None, rewrite=REWRITES) -> Retriever:
    docs = docs if docs is not None else make_docs("alpha text", "beta text")
    return Retriever(
        client=FakeClient(rewrite),
        vectorstore=FakeVectorStore(docs),
        chunks=docs,
        bm25=FakeBM25(bm25_scores if bm25_scores is not None else [0.0] * len(docs)),
        reranker=FakeReranker(),
        settings=settings,
    )


@pytest.fixture
def admin(settings):
    return AccessPolicy.for_role("admin", settings)


# ── query rewriting ──────────────────────────────────────────────


def test_rewrite_query_parses_numbered_list(settings):
    retriever = build(settings)
    assert retriever.rewrite_query("original") == [
        "alternative one",
        "alternative two",
        "alternative three",
    ]


def test_rewrite_query_uses_configured_model(settings):
    retriever = build(settings)
    retriever.rewrite_query("original")
    assert retriever.client.calls[0]["model"] == settings.rag.llm_model


# ── hybrid search ────────────────────────────────────────────────


def test_search_includes_the_original_question(settings, admin):
    retriever = build(settings)
    retriever.search("my question", k=2, policy=admin)
    assert "my question" in [q for q, _ in retriever.vectorstore.queries]


def test_search_runs_every_variation(settings, admin):
    retriever = build(settings)
    retriever.search("my question", k=2, policy=admin)
    # 3 rewrites + the original
    assert len(retriever.vectorstore.queries) == 4


def test_search_honours_k(settings, admin):
    retriever = build(settings)
    retriever.search("q", k=1, policy=admin)
    assert {k for _, k in retriever.vectorstore.queries} == {1}


def test_search_deduplicates_repeated_hits(settings, admin):
    docs = make_docs("shared chunk", "other chunk")
    retriever = build(settings, docs=docs)
    results = retriever.search("q", k=2, policy=admin)
    assert len(results) == 2  # not 8, despite 4 query variations


def test_bm25_hits_are_merged_in(settings, admin):
    docs = make_docs("vector only", "keyword only")
    retriever = Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore(docs[:1]),  # vector search sees only the first
        chunks=docs,
        bm25=FakeBM25([0.0, 5.0]),              # BM25 scores only the second
        reranker=FakeReranker(),
        settings=settings,
    )
    contents = {d.page_content for d in retriever.search("q", k=1, policy=admin)}
    assert contents == {"vector only", "keyword only"}


def test_zero_scoring_bm25_hits_are_ignored(settings, admin):
    docs = make_docs("vector only", "never matched")
    retriever = Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore(docs[:1]),
        chunks=docs,
        bm25=FakeBM25([0.0, 0.0]),
        reranker=FakeReranker(),
        settings=settings,
    )
    assert [d.page_content for d in retriever.search("q", k=1, policy=admin)] == [
        "vector only"
    ]


# ── reranking ────────────────────────────────────────────────────


def test_rerank_orders_by_score_and_truncates(settings):
    retriever = build(settings)
    docs = make_docs("nothing relevant", "torque spec value", "torque spec value here")
    ranked = retriever.rerank("torque spec", docs, top_n=2)
    assert len(ranked) == 2
    assert "torque" in ranked[0].page_content


def test_rerank_handles_empty_input(settings):
    assert build(settings).rerank("q", [], top_n=3) == []


def test_rerank_top_n_larger_than_input_is_safe(settings):
    retriever = build(settings)
    docs = make_docs("one")
    assert len(retriever.rerank("one", docs, top_n=10)) == 1


# ── full pipeline ────────────────────────────────────────────────


def test_retrieve_respects_role_depth(settings):
    docs = make_docs("torque a", "torque b", "torque c", "unrelated")
    retriever = build(settings, docs=docs)
    perms = settings.permissions_for("viewer")

    result = retriever.retrieve(
        "torque",
        max_results=perms.max_results,
        top_n=perms.top_n_rerank,
        policy=AccessPolicy.for_role("viewer", settings),
    )
    assert len(result) == perms.top_n_rerank == 2


def test_admin_gets_more_context_than_viewer(settings):
    docs = make_docs(*[f"torque chunk {i}" for i in range(8)])
    retriever = build(settings, docs=docs)

    def depth(role: str) -> int:
        perms = settings.permissions_for(role)
        return len(
            retriever.retrieve(
                "torque",
                max_results=perms.max_results,
                top_n=perms.top_n_rerank,
                policy=AccessPolicy.for_role(role, settings),
            )
        )

    assert depth("admin") > depth("viewer")


def test_chunk_count_reports_index_size(settings):
    retriever = build(settings, docs=make_docs("a", "b", "c"))
    assert retriever.chunk_count == 3


def test_empty_index_yields_no_results(settings, admin):
    retriever = Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore([]),
        chunks=[],
        bm25=FakeBM25([]),
        reranker=FakeReranker(),
        settings=settings,
    )
    assert retriever.chunk_count == 0
    assert retriever.retrieve("q", max_results=5, top_n=3, policy=admin) == []
