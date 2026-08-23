"""
Ingestion pipeline.

The pipeline used to live inside scripts/ingest.py, reachable only from a
terminal — which meant a deployed instance could never load a document, and
none of this logic could be tested. These tests exercise it directly.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.ingestion import (
    IngestionError,
    chunk_pages,
    has_extractable_text,
    ingest_document,
    ingest_upload,
    roles_cleared_for,
)
from rag.settings import UnknownTenantError


def page(text: str, number: int = 1) -> Document:
    return Document(page_content=text, metadata={"page": number})


# ── clearance preview ────────────────────────────────────────────


def test_roles_cleared_for_public(settings):
    assert roles_cleared_for(settings, "public") == ("admin", "support", "viewer")


def test_roles_cleared_for_confidential(settings):
    assert roles_cleared_for(settings, "confidential") == ("admin",)


def test_roles_cleared_for_unknown_label_is_empty(settings):
    """Surfaced in the UI as "nobody could retrieve this"."""
    assert roles_cleared_for(settings, "nonsense") == ()


# ── OCR heuristic ────────────────────────────────────────────────


def test_pages_with_real_text_skip_ocr():
    assert has_extractable_text([page("x" * 200), page("y" * 200)])


def test_mostly_blank_pages_trigger_ocr():
    assert not has_extractable_text([page(""), page(""), page("y" * 200)])


def test_no_pages_is_not_extractable():
    """Guards against dividing by zero on an unreadable file."""
    assert not has_extractable_text([])


def test_short_pages_count_as_blank():
    assert not has_extractable_text([page("hi"), page("yo")])


# ── chunking ─────────────────────────────────────────────────────


def test_chunking_splits_long_pages():
    chunks = chunk_pages([page("word " * 500)], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunking_preserves_page_metadata():
    chunks = chunk_pages([page("word " * 500, number=7)], 200, 20)
    assert all(c.metadata["page"] == 7 for c in chunks)


# ── validation happens before any paid work ──────────────────────


def test_missing_file_is_rejected(settings, tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        ingest_document(
            tmp_path / "absent.pdf",
            settings=settings,
            tenant_id="acme",
            api_key="sk-test",
        )


def test_unknown_tenant_is_rejected(settings, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(UnknownTenantError):
        ingest_document(
            pdf, settings=settings, tenant_id="nowhere", api_key="sk-test"
        )


def test_unknown_classification_is_rejected(settings, tmp_path):
    """Rejected before embedding, so a typo costs nothing."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(IngestionError, match="Unknown classification"):
        ingest_document(
            pdf,
            settings=settings,
            tenant_id="acme",
            api_key="sk-test",
            classification="top-secret",
        )


def test_missing_api_key_is_rejected(settings, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(IngestionError, match="API key"):
        ingest_document(pdf, settings=settings, tenant_id="acme", api_key="")


def test_empty_upload_is_rejected(settings):
    with pytest.raises(IngestionError, match="empty"):
        ingest_upload(
            b"", "x.pdf", settings=settings, tenant_id="acme", api_key="sk-test"
        )


# ── upload staging ───────────────────────────────────────────────


def test_upload_temp_file_is_always_removed(settings, monkeypatch):
    """
    An uploaded document must not be left on disk after ingestion. The temp
    file is cleaned up even when the pipeline raises.
    """
    seen: list[str] = []

    def capture_and_fail(path, **kwargs):
        seen.append(str(path))
        raise IngestionError("boom")

    monkeypatch.setattr("rag.ingestion.ingest_document", capture_and_fail)

    with pytest.raises(IngestionError, match="boom"):
        ingest_upload(
            b"%PDF-1.4 data",
            "handbook.pdf",
            settings=settings,
            tenant_id="acme",
            api_key="sk-test",
        )

    assert seen, "ingest_document was never reached"
    import os

    assert not os.path.exists(seen[0]), "temp upload file was left behind"


def test_upload_defaults_doc_id_and_title_from_filename(settings, monkeypatch):
    captured: dict = {}

    def capture(path, **kwargs):
        captured.update(kwargs)
        raise IngestionError("stop here")

    monkeypatch.setattr("rag.ingestion.ingest_document", capture)

    with pytest.raises(IngestionError):
        ingest_upload(
            b"%PDF-1.4",
            "Employee Handbook.pdf",
            settings=settings,
            tenant_id="acme",
            api_key="sk-test",
        )

    assert captured["doc_id"] == "employee-handbook"
    assert captured["title"] == "Employee Handbook.pdf"


def test_upload_respects_an_explicit_doc_id(settings, monkeypatch):
    captured: dict = {}

    def capture(path, **kwargs):
        captured.update(kwargs)
        raise IngestionError("stop here")

    monkeypatch.setattr("rag.ingestion.ingest_document", capture)

    with pytest.raises(IngestionError):
        ingest_upload(
            b"%PDF-1.4",
            "whatever.pdf",
            settings=settings,
            tenant_id="acme",
            api_key="sk-test",
            doc_id="chosen-id",
        )

    assert captured["doc_id"] == "chosen-id"
