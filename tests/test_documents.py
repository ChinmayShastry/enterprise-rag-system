"""
Document identity and lifecycle.

The headline behaviour: re-ingesting a document replaces it. Ingestion used to
call Chroma.from_documents(), which mints a fresh UUID per chunk, so running it
twice on the same PDF left two full copies in the index.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.documents import (
    DOC_ID_KEY,
    TENANT_KEY,
    TITLE_KEY,
    chunk_id,
    delete_document,
    doc_id_from_path,
    list_documents,
    prepare_chunks,
    slugify,
)


class FakeCollection:
    """
    Minimal stand-in for a Chroma collection with upsert-by-id semantics,
    which is what stable chunk IDs rely on.
    """

    def __init__(self):
        self.store: dict[str, Document] = {}

    def add_documents(self, documents, ids):
        for doc, id_ in zip(documents, ids):
            self.store[id_] = doc

    def get(self, where=None):
        items = list(self.store.items())
        if where:
            key, value = next(iter(where.items()))
            items = [(i, d) for i, d in items if d.metadata.get(key) == value]
        return {
            "ids": [i for i, _ in items],
            "documents": [d.page_content for _, d in items],
            "metadatas": [d.metadata for _, d in items],
        }

    def delete(self, ids):
        for id_ in ids:
            self.store.pop(id_, None)


def pages(*texts: str) -> list[Document]:
    return [Document(page_content=t, metadata={"page": i + 1}) for i, t in enumerate(texts)]


def ingest(store, texts, *, tenant="acme", doc_id="manual", title="Manual",
           classification="internal"):
    chunks, ids = prepare_chunks(
        pages(*texts),
        tenant=tenant,
        doc_id=doc_id,
        title=title,
        classification=classification,
    )
    delete_document(store, doc_id)
    store.add_documents(chunks, ids)
    return chunks, ids


# ── identity ─────────────────────────────────────────────────────


def test_slugify_normalises_titles():
    assert slugify("Bosch Dishwasher Manual v2.1") == "bosch-dishwasher-manual-v2-1"


def test_slugify_never_returns_empty():
    assert slugify("!!!") == "document"


def test_doc_id_derives_from_filename():
    assert doc_id_from_path("data/Employee Handbook.pdf") == "employee-handbook"


def test_chunk_id_is_deterministic():
    assert chunk_id("acme", "manual", 0, "text") == chunk_id("acme", "manual", 0, "text")


def test_chunk_id_varies_by_tenant():
    """Identical documents ingested by two tenants must not collide."""
    assert chunk_id("acme", "manual", 0, "t") != chunk_id("globex", "manual", 0, "t")


def test_chunk_id_varies_by_document_and_position():
    assert chunk_id("acme", "a", 0, "t") != chunk_id("acme", "b", 0, "t")
    assert chunk_id("acme", "a", 0, "t") != chunk_id("acme", "a", 1, "t")


def test_chunk_id_varies_by_content():
    """Edited text gets a new ID rather than overwriting under a reused slot."""
    assert chunk_id("acme", "a", 0, "old") != chunk_id("acme", "a", 0, "new")


def test_prepare_chunks_stamps_all_metadata():
    chunks, ids = prepare_chunks(
        pages("one", "two"),
        tenant="acme", doc_id="manual", title="Manual", classification="internal",
    )
    assert len(ids) == 2
    for chunk in chunks:
        assert chunk.metadata[TENANT_KEY] == "acme"
        assert chunk.metadata[DOC_ID_KEY] == "manual"
        assert chunk.metadata[TITLE_KEY] == "Manual"
        assert chunk.metadata["classification"] == "internal"
        assert chunk.metadata["page"]  # original page survives


# ── idempotent re-ingestion ──────────────────────────────────────


def test_reingesting_the_same_document_does_not_duplicate():
    """The bug this whole module exists to fix."""
    store = FakeCollection()
    ingest(store, ["alpha", "beta", "gamma"])
    assert len(store.store) == 3

    ingest(store, ["alpha", "beta", "gamma"])
    assert len(store.store) == 3


def test_reingesting_a_shorter_version_drops_the_old_tail():
    """
    v2 with fewer chunks must not leave v1's tail behind, where it would keep
    being retrieved as though it were current.
    """
    store = FakeCollection()
    ingest(store, ["alpha", "beta", "gamma"])
    ingest(store, ["alpha only"])

    assert len(store.store) == 1
    assert [d.page_content for d in store.store.values()] == ["alpha only"]


def test_reingesting_edited_content_replaces_it():
    store = FakeCollection()
    ingest(store, ["torque is 30 Nm"])
    ingest(store, ["torque is 34 Nm"])

    contents = [d.page_content for d in store.store.values()]
    assert contents == ["torque is 34 Nm"]


def test_two_documents_coexist():
    store = FakeCollection()
    ingest(store, ["manual text"], doc_id="manual", title="Manual")
    ingest(store, ["policy text"], doc_id="policy", title="Policy")
    assert len(store.store) == 2


def test_same_document_in_two_tenants_lands_in_separate_collections():
    """
    Each tenant has its own collection, so the same PDF ingested by both is
    stored twice over — under different IDs, in different stores. Deleting one
    tenant's copy cannot touch the other's.
    """
    acme, globex = FakeCollection(), FakeCollection()
    _, acme_ids = ingest(acme, ["shared text"], tenant="acme")
    _, globex_ids = ingest(globex, ["shared text"], tenant="globex")

    assert acme_ids != globex_ids
    assert len(acme.store) == len(globex.store) == 1

    delete_document(acme, "manual")
    assert acme.store == {}
    assert len(globex.store) == 1


# ── inventory ────────────────────────────────────────────────────


def test_list_documents_groups_and_counts_chunks():
    store = FakeCollection()
    ingest(store, ["a", "b", "c"], doc_id="manual", title="Manual")
    ingest(store, ["x"], doc_id="policy", title="Policy", classification="confidential")

    docs = list_documents(store)
    assert [(d.doc_id, d.chunk_count) for d in docs] == [("manual", 3), ("policy", 1)]
    assert docs[1].classification == "confidential"
    assert docs[0].title == "Manual"


def test_list_documents_is_empty_for_a_fresh_collection():
    assert list_documents(FakeCollection()) == []


def test_list_documents_skips_chunks_without_a_doc_id():
    """Legacy chunks predating document identity are not inventable."""
    store = FakeCollection()
    store.store["legacy"] = Document(page_content="old", metadata={"page": 1})
    assert list_documents(store) == []


def test_delete_document_removes_only_that_document():
    store = FakeCollection()
    ingest(store, ["a", "b"], doc_id="manual")
    ingest(store, ["x"], doc_id="policy")

    removed = delete_document(store, "manual")
    assert removed == 2
    assert [d.doc_id for d in list_documents(store)] == ["policy"]


def test_delete_missing_document_reports_zero():
    assert delete_document(FakeCollection(), "nope") == 0
