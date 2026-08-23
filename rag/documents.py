"""
rag/documents.py
Document identity and lifecycle within a tenant's collection.

Ingestion used to call Chroma.from_documents(), which mints a fresh UUID per
chunk. Re-running it on the same PDF therefore appended a second full copy of
the corpus, so every answer was drawn from a silently duplicated index. Chunk
IDs are now derived from tenant, document and content, which makes re-ingesting
a document replace it rather than duplicate it.

Metadata written per chunk:
    tenant          owning tenant id
    doc_id          stable document identifier
    title           human-readable document name
    classification  security label (see rag/access.py)
    page            1-based page number
    source          original filename
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

TENANT_KEY = "tenant"
DOC_ID_KEY = "doc_id"
TITLE_KEY = "title"


# ─────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────


def slugify(value: str) -> str:
    """Turn a filename or title into a stable, filesystem-agnostic doc_id."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def doc_id_from_path(pdf_path: str | Path) -> str:
    return slugify(Path(pdf_path).stem)


def chunk_id(tenant: str, doc_id: str, index: int, content: str) -> str:
    """
    Deterministic ID for one chunk.

    Includes the content, so an edited page produces a new ID rather than
    silently overwriting the old text under a reused position. Includes the
    tenant, so identical documents ingested by two tenants never collide.
    """
    payload = "\x00".join([tenant, doc_id, str(index), content])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def prepare_chunks(
    chunks: list[Document],
    *,
    tenant: str,
    doc_id: str,
    title: str,
    classification: str,
) -> tuple[list[Document], list[str]]:
    """
    Stamp identity and security metadata onto chunks and derive their IDs.

    Returns the chunks alongside the ID list to pass to the vector store.
    """
    ids = []
    for index, chunk in enumerate(chunks):
        chunk.metadata[TENANT_KEY] = tenant
        chunk.metadata[DOC_ID_KEY] = doc_id
        chunk.metadata[TITLE_KEY] = title
        chunk.metadata["classification"] = classification
        ids.append(chunk_id(tenant, doc_id, index, chunk.page_content))
    return chunks, ids


# ─────────────────────────────────────────────────────────────────
# Inventory
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    title: str
    classification: str
    chunk_count: int

    @property
    def label(self) -> str:
        return f"{self.title} ({self.doc_id})"


def list_documents(vectorstore) -> list[DocumentInfo]:
    """
    Summarise the documents in a tenant's collection.

    The vector store is already tenant-scoped, so this needs no tenant filter.
    """
    stored = vectorstore.get()
    seen: dict[str, dict] = {}

    for meta in stored["metadatas"]:
        meta = meta or {}
        doc_id = meta.get(DOC_ID_KEY)
        if not doc_id:
            continue
        entry = seen.setdefault(
            doc_id,
            {
                "title": meta.get(TITLE_KEY, doc_id),
                "classification": meta.get("classification", "unlabelled"),
                "count": 0,
            },
        )
        entry["count"] += 1

    return sorted(
        (
            DocumentInfo(
                doc_id=doc_id,
                title=entry["title"],
                classification=entry["classification"],
                chunk_count=entry["count"],
            )
            for doc_id, entry in seen.items()
        ),
        key=lambda d: d.doc_id,
    )


def delete_document(vectorstore, doc_id: str) -> int:
    """
    Remove every chunk of one document. Returns how many were deleted.

    This is what makes re-ingestion a replace: a new version of a document may
    produce fewer chunks than the old one, and without an explicit delete the
    leftover tail would linger and keep being retrieved.
    """
    stored = vectorstore.get(where={DOC_ID_KEY: doc_id})
    ids = stored.get("ids") or []
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)
