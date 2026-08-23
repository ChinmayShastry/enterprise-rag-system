"""
rag/ingestion.py
The document ingestion pipeline: extract → chunk → label → embed → store.

This logic used to live inside scripts/ingest.py, which meant it could only be
driven from a terminal. A deployed instance — Streamlit Cloud gives you no
shell — had no way to load a document at all. The pipeline now lives here, free
of both argparse and Streamlit, so the CLI and the admin upload page run the
same code rather than two drifting copies.

Progress is reported through an `on_progress` callback so each caller can
render it appropriately: the CLI prints, the UI updates a status widget, and
tests pass nothing.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from rag.documents import (
    delete_document,
    doc_id_from_path,
    prepare_chunks,
    slugify,
)
from rag.settings import Settings, TenantConfig

Progress = Callable[[str], None]


class IngestionError(RuntimeError):
    """Raised when a document cannot be ingested as requested."""


@dataclass(frozen=True)
class IngestionResult:
    tenant_id: str
    doc_id: str
    title: str
    classification: str
    page_count: int
    chunk_count: int
    replaced_chunks: int
    used_ocr: bool
    collection_total: int
    cleared_roles: tuple[str, ...]

    @property
    def replaced(self) -> bool:
        return self.replaced_chunks > 0


def _noop(_message: str) -> None:
    pass


def roles_cleared_for(settings: Settings, classification: str) -> tuple[str, ...]:
    """
    Which roles will be able to retrieve a document with this label.

    Surfaced before ingestion so an operator can see that picking
    'confidential' means a viewer will never see the document — rather than
    discovering it later as an apparent retrieval failure.
    """
    return tuple(
        sorted(
            name
            for name, perms in settings.roles.items()
            if classification in perms.clearance
        )
    )


# ─────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────


def extract_text_pypdf(pdf_path: str | Path) -> list[Document]:
    """Extract text from a text-based PDF."""
    from langchain_community.document_loaders import PyPDFLoader

    pages = PyPDFLoader(str(pdf_path)).load()
    for index, page in enumerate(pages):
        # PyPDF's own page metadata is 0-based and occasionally absent; the
        # citations shown to users are 1-based, so normalise here.
        page.metadata["page"] = index + 1
        page.metadata["source"] = Path(pdf_path).name
    return pages


def extract_text_ocr(
    pdf_path: str | Path, on_progress: Progress = _noop
) -> list[Document]:
    """Extract text from a scanned PDF with Tesseract."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        raise IngestionError(
            "OCR dependencies are missing. Install pytesseract and pdf2image, "
            "plus the tesseract-ocr and poppler-utils system packages."
        ) from e

    on_progress("Running OCR — this can take several minutes…")
    images = convert_from_path(str(pdf_path), dpi=200)

    pages = []
    for index, image in enumerate(images):
        pages.append(
            Document(
                page_content=pytesseract.image_to_string(image),
                metadata={"page": index + 1, "source": Path(pdf_path).name},
            )
        )
        if (index + 1) % 10 == 0:
            on_progress(f"OCR: {index + 1}/{len(images)} pages")
    return pages


def has_extractable_text(pages: list[Document]) -> bool:
    """Whether enough pages carry real text to skip OCR."""
    if not pages:
        return False
    substantial = sum(1 for p in pages if len(p.page_content.strip()) > 50)
    return substantial / len(pages) > 0.5


def extract_pages(
    pdf_path: str | Path,
    *,
    force_ocr: bool = False,
    on_progress: Progress = _noop,
) -> tuple[list[Document], bool]:
    """
    Pull text out of a PDF, falling back to OCR when it is mostly images.
    Returns the pages and whether OCR was used.
    """
    if force_ocr:
        return extract_text_ocr(pdf_path, on_progress), True

    pages = extract_text_pypdf(pdf_path)
    if has_extractable_text(pages):
        on_progress(f"Extracted text from {len(pages)} pages")
        return pages, False

    on_progress("Most pages look empty — switching to OCR")
    return extract_text_ocr(pdf_path, on_progress), True


# ─────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────


def chunk_pages(
    pages: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(pages)


# ─────────────────────────────────────────────────────────────────
# The pipeline
# ─────────────────────────────────────────────────────────────────


def _open_collection(settings: Settings, tenant: TenantConfig, api_key: str):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    return Chroma(
        collection_name=tenant.collection_name,
        embedding_function=OpenAIEmbeddings(
            model=settings.rag.embedding_model, api_key=api_key
        ),
        persist_directory=str(settings.rag.chroma_path),
    )


def ingest_document(
    pdf_path: str | Path,
    *,
    settings: Settings,
    tenant_id: str,
    api_key: str,
    doc_id: str | None = None,
    title: str | None = None,
    classification: str | None = None,
    force_ocr: bool = False,
    reset: bool = False,
    on_progress: Progress = _noop,
) -> IngestionResult:
    """
    Ingest one PDF into one tenant's collection.

    Everything is validated before any embedding call, so a mistyped tenant or
    classification costs nothing rather than failing halfway through a paid
    run.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise IngestionError(f"File not found: {path}")

    tenant = settings.tenant(tenant_id)  # raises UnknownTenantError

    label = classification or settings.rag.default_classification
    if label not in settings.classifications:
        raise IngestionError(
            f"Unknown classification '{label}'. Declared classifications are: "
            f"{', '.join(settings.classifications)}"
        )

    if not api_key:
        raise IngestionError("An OpenAI API key is required to embed a document.")

    resolved_doc_id = slugify(doc_id) if doc_id else doc_id_from_path(path)
    resolved_title = title or path.name

    pages, used_ocr = extract_pages(path, force_ocr=force_ocr, on_progress=on_progress)
    if not pages:
        raise IngestionError("No pages could be read from this file.")

    chunks = chunk_pages(pages, settings.rag.chunk_size, settings.rag.chunk_overlap)
    if not chunks:
        raise IngestionError(
            "The document produced no text chunks. If it is a scanned PDF, retry "
            "with OCR forced."
        )
    on_progress(f"{len(pages)} pages → {len(chunks)} chunks")

    chunks, ids = prepare_chunks(
        chunks,
        tenant=tenant.tenant_id,
        doc_id=resolved_doc_id,
        title=resolved_title,
        classification=label,
    )

    if reset:
        on_progress(f"Wiping collection '{tenant.collection_name}'")
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.rag.chroma_path))
        try:
            client.delete_collection(tenant.collection_name)
        except Exception:
            pass  # Nothing to wipe is a normal first run, not an error.

    store = _open_collection(settings, tenant, api_key)

    # Replace rather than append. A new version may produce fewer chunks than
    # the old one, and the leftover tail would otherwise linger in the index
    # and keep being retrieved as though current.
    replaced = 0 if reset else delete_document(store, resolved_doc_id)
    if replaced:
        on_progress(f"Replacing '{resolved_doc_id}': removed {replaced} old chunks")

    on_progress(f"Embedding {len(chunks)} chunks with {settings.rag.embedding_model}…")
    store.add_documents(documents=chunks, ids=ids)

    try:
        total = store._collection.count()
    except Exception:
        total = len(chunks)

    on_progress("Done")

    return IngestionResult(
        tenant_id=tenant.tenant_id,
        doc_id=resolved_doc_id,
        title=resolved_title,
        classification=label,
        page_count=len(pages),
        chunk_count=len(chunks),
        replaced_chunks=replaced,
        used_ocr=used_ocr,
        collection_total=total,
        cleared_roles=roles_cleared_for(settings, label),
    )


def ingest_upload(
    data: bytes,
    filename: str,
    **kwargs,
) -> IngestionResult:
    """
    Ingest an in-memory upload.

    A browser upload arrives as bytes but PyPDFLoader and pdf2image both want a
    real path, so the bytes are staged in a temp file that is always removed —
    an uploaded document must not be left lying around on the server after the
    vectors have been written.
    """
    if not data:
        raise IngestionError("The uploaded file is empty.")

    suffix = Path(filename).suffix or ".pdf"
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(data)
        kwargs.setdefault("title", filename)
        kwargs.setdefault("doc_id", slugify(Path(filename).stem))
        return ingest_document(temp_path, **kwargs)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
