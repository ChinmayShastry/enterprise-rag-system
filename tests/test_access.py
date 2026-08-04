"""
Document-level access control.

The load-bearing tests are at the bottom: a viewer must not be able to reach
confidential text through the vector path, through the BM25 path, or through
a reranker that would happily score it highly.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.access import AccessPolicy, unlabelled
from rag.retrieval import Retriever
from rag.settings import ConfigError, load_settings
from tests.test_retrieval import (
    REWRITES,
    FakeBM25,
    FakeClient,
    FakeReranker,
    FakeVectorStore,
)


def doc(text: str, classification: str | None = "public") -> Document:
    metadata = {"page": 1}
    if classification is not None:
        metadata["classification"] = classification
    return Document(page_content=text, metadata=metadata)


# ── policy basics ────────────────────────────────────────────────


def test_policy_permits_cleared_classification(settings):
    policy = AccessPolicy.for_role("support", settings)
    assert policy.permits(doc("x", "public"))
    assert policy.permits(doc("x", "internal"))


def test_policy_denies_uncleared_classification(settings):
    policy = AccessPolicy.for_role("support", settings)
    assert not policy.permits(doc("x", "confidential"))


def test_policy_denies_unlabelled_chunks(settings):
    """Deny-by-default: no label means nobody, not everybody."""
    admin = AccessPolicy.for_role("admin", settings)
    assert not admin.permits(doc("x", classification=None))


def test_policy_denies_unrecognised_label(settings):
    admin = AccessPolicy.for_role("admin", settings)
    assert not admin.permits(doc("x", "top-secret-typo"))


def test_policy_denies_non_string_label(settings):
    admin = AccessPolicy.for_role("admin", settings)
    assert not admin.permits(Document(page_content="x", metadata={"classification": 1}))


def test_unknown_role_is_cleared_for_nothing(settings):
    policy = AccessPolicy.for_role("no-such-role", settings)
    assert policy.denies_everything
    assert policy.filter([doc("x", "public")]) == []


def test_where_clause_lists_only_cleared_labels(settings):
    policy = AccessPolicy.for_role("viewer", settings)
    assert policy.where_clause() == {"classification": {"$in": ["public"]}}


def test_where_clause_is_none_when_cleared_for_nothing(settings):
    """
    None must mean "retrieve nothing" to callers. Passing it to Chroma as a
    filter would mean "no filter" — i.e. return everything — so search() has
    to short-circuit before it gets there.
    """
    assert AccessPolicy.for_role("suspended", settings).where_clause() is None


def test_filter_preserves_order(settings):
    policy = AccessPolicy.for_role("support", settings)
    docs = [doc("a", "public"), doc("b", "confidential"), doc("c", "internal")]
    assert [d.page_content for d in policy.filter(docs)] == ["a", "c"]


def test_unlabelled_finds_chunks_missing_a_label():
    docs = [doc("a", "public"), doc("b", classification=None)]
    assert [d.page_content for d in unlabelled(docs)] == ["b"]


# ── config validation ────────────────────────────────────────────


def test_clearance_typo_is_rejected_at_load(tmp_path, config_file):
    import yaml

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["roles"]["viewer"]["clearance"] = ["publik"]  # typo
    bad = tmp_path / "typo.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown classification"):
        load_settings(bad)


def test_queryable_role_with_no_clearance_is_rejected(tmp_path, config_file):
    import yaml

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["roles"]["viewer"]["clearance"] = []
    bad = tmp_path / "empty.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="empty clearance"):
        load_settings(bad)


def test_default_classification_must_be_declared(tmp_path, config_file):
    import yaml

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["rag"]["default_classification"] = "nonsense"
    bad = tmp_path / "bad_default.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="default_classification"):
        load_settings(bad)


# ── end-to-end enforcement ───────────────────────────────────────

MIXED = [
    Document(
        page_content="public: how to change the filter",
        metadata={"page": 1, "classification": "public"},
    ),
    Document(
        page_content="confidential: severance terms and payout schedule",
        metadata={"page": 2, "classification": "confidential"},
    ),
]


def build_mixed(settings, bm25_scores):
    return Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore(MIXED),
        chunks=MIXED,
        bm25=FakeBM25(bm25_scores),
        reranker=FakeReranker(),
        settings=settings,
    )


def retrieve_as(retriever, role, settings, question="severance terms"):
    perms = settings.permissions_for(role)
    return retriever.retrieve(
        question,
        max_results=perms.max_results,
        top_n=perms.top_n_rerank,
        policy=AccessPolicy.for_role(role, settings),
    )


def test_viewer_cannot_retrieve_confidential_via_vector_search(settings):
    # BM25 contributes nothing, so anything returned came from the vector path.
    retriever = build_mixed(settings, [0.0, 0.0])
    results = retrieve_as(retriever, "viewer", settings)
    assert all(d.metadata["classification"] == "public" for d in results)
    assert not any("severance" in d.page_content for d in results)


def test_viewer_cannot_retrieve_confidential_via_bm25(settings):
    # BM25 scores the confidential chunk highest — the exact leak path a
    # metadata filter on the vector store alone would miss.
    retriever = build_mixed(settings, [0.0, 9.9])
    results = retrieve_as(retriever, "viewer", settings)
    assert not any("severance" in d.page_content for d in results)


def search_as(retriever, role, settings, question="severance terms"):
    """
    Call search() rather than retrieve(), so the post-retrieval backstop is not
    in play. These assertions fail if either search path stops enforcing
    clearance, which retrieve()-level tests would mask.
    """
    perms = settings.permissions_for(role)
    return retriever.search(
        question,
        k=perms.max_results,
        policy=AccessPolicy.for_role(role, settings),
    )


def test_bm25_path_excludes_denied_chunks_before_the_backstop(settings):
    retriever = build_mixed(settings, [0.0, 9.9])
    assert not any("severance" in d.page_content for d in search_as(retriever, "viewer", settings))


def test_vector_path_excludes_denied_chunks_before_the_backstop(settings):
    retriever = build_mixed(settings, [0.0, 0.0])
    assert not any("severance" in d.page_content for d in search_as(retriever, "viewer", settings))


def test_denied_chunks_do_not_crowd_out_permitted_ones(settings):
    """
    A denied BM25 hit must be skipped rather than consume one of the k slots,
    otherwise a viewer's context silently shrinks as confidential content grows.
    """
    docs = [
        Document(page_content=f"secret {i}", metadata={"page": i, "classification": "confidential"})
        for i in range(5)
    ] + [
        Document(page_content="public answer", metadata={"page": 9, "classification": "public"})
    ]
    retriever = Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore([]),          # isolate the BM25 path
        chunks=docs,
        bm25=FakeBM25([9.0, 8.0, 7.0, 6.0, 5.0, 1.0]),  # public chunk ranks last
        reranker=FakeReranker(),
        settings=settings,
    )
    found = retriever.search(
        "answer", k=2, policy=AccessPolicy.for_role("viewer", settings)
    )
    assert [d.page_content for d in found] == ["public answer"]


def test_vector_search_is_filtered_server_side_not_after_the_fact(settings):
    """
    The where-clause must reach the vector store. Filtering only afterwards
    would still be correct, but a viewer's top-k would be full of chunks they
    cannot see, starving the answer of usable context.
    """
    retriever = build_mixed(settings, [0.0, 0.0])
    retrieve_as(retriever, "viewer", settings)
    assert retriever.vectorstore.filters
    assert all(
        f == {"classification": {"$in": ["public"]}}
        for f in retriever.vectorstore.filters
    )


def test_admin_can_retrieve_confidential(settings):
    retriever = build_mixed(settings, [0.0, 9.9])
    results = retrieve_as(retriever, "admin", settings)
    assert any("severance" in d.page_content for d in results)


def test_support_sits_between_viewer_and_admin(settings):
    retriever = build_mixed(settings, [0.0, 9.9])
    results = retrieve_as(retriever, "support", settings)
    assert not any("severance" in d.page_content for d in results)


def test_role_denied_all_clearance_retrieves_nothing(settings):
    retriever = build_mixed(settings, [9.9, 9.9])
    assert retrieve_as(retriever, "suspended", settings) == []


def test_denied_role_never_touches_the_vector_store(settings):
    """Short-circuit before querying, so a None filter can't mean 'no filter'."""
    retriever = build_mixed(settings, [9.9, 9.9])
    retrieve_as(retriever, "suspended", settings)
    assert retriever.vectorstore.queries == []


def test_backstop_filters_a_leak_from_a_broken_search_path(settings):
    """
    Both search paths already enforce clearance, so removing the final
    policy.filter() breaks no other test. This one simulates a regression in
    search() to prove the backstop is load-bearing rather than decoration.
    """

    class LeakyRetriever(Retriever):
        def search(self, question, k, *, policy):
            return list(MIXED)  # ignores clearance entirely

    leaky = LeakyRetriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore(MIXED),
        chunks=MIXED,
        bm25=FakeBM25([0.0, 0.0]),
        reranker=FakeReranker(),
        settings=settings,
    )
    results = retrieve_as(leaky, "viewer", settings)
    assert not any("severance" in d.page_content for d in results)


def test_unlabelled_chunks_are_invisible_to_every_role(settings):
    legacy = [Document(page_content="legacy unlabelled text", metadata={"page": 1})]
    retriever = Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore(legacy),
        chunks=legacy,
        bm25=FakeBM25([9.9]),
        reranker=FakeReranker(),
        settings=settings,
    )
    assert retriever.unlabelled_count == 1
    for role in ("admin", "support", "viewer"):
        assert retrieve_as(retriever, role, settings, "legacy") == []
