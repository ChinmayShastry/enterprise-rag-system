#!/usr/bin/env python3
"""
scripts/ingest.py
Ingest a PDF into ChromaDB for the RAG system.

Usage:
    python scripts/ingest.py --pdf data/your_document.pdf
    python scripts/ingest.py --pdf data/your_document.pdf --ocr
    python scripts/ingest.py --pdf data/your_document.pdf --reset

Options:
    --pdf PATH       Path to the PDF file (required)
    --ocr            Force OCR even for text-based PDFs
    --reset          Wipe the existing ChromaDB collection before ingesting
    --config PATH    Path to config.yaml (default: config/config.yaml)
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.settings import ConfigError, Settings, load_settings

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--ocr", action="store_true", help="Force OCR extraction")
    parser.add_argument("--reset", action="store_true", help="Wipe existing collection first")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    return parser.parse_args()


def extract_text_pypdf(pdf_path: str) -> list:
    """Extract text using PyPDF (works for text-based PDFs)."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # Ensure metadata has 'page' as 1-based int
    for i, page in enumerate(pages):
        page.metadata["page"] = i + 1
        page.metadata["source"] = str(Path(pdf_path).name)
    return pages


def extract_text_ocr(pdf_path: str) -> list:
    """Extract text using Tesseract OCR (required for scanned PDFs)."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from langchain_core.documents import Document
    except ImportError:
        print("❌  OCR dependencies missing. Run: pip install pytesseract pdf2image")
        sys.exit(1)

    print("🔄  Running OCR (this may take a few minutes)…")
    images = convert_from_path(pdf_path, dpi=200)
    pages = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        pages.append(
            Document(
                page_content=text,
                metadata={"page": i + 1, "source": str(Path(pdf_path).name)},
            )
        )
        if (i + 1) % 10 == 0:
            print(f"   ✅  OCR: {i + 1}/{len(images)} pages processed…")
    return pages


def has_text(pages: list) -> bool:
    """Heuristic: if most pages have content, skip OCR."""
    non_empty = sum(1 for p in pages if len(p.page_content.strip()) > 50)
    return non_empty / max(len(pages), 1) > 0.5


def chunk_documents(pages: list, chunk_size: int, chunk_overlap: int) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    print(f"   ✅  {len(pages)} pages → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def embed_and_store(chunks: list, settings: Settings, api_key: str, reset: bool) -> None:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    rag_cfg = settings.rag
    chroma_path = str(rag_cfg.chroma_path)

    embedding_model = OpenAIEmbeddings(model=rag_cfg.embedding_model, api_key=api_key)

    if reset:
        print("🗑️   Wiping existing collection…")
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            client.delete_collection(rag_cfg.collection_name)
            print("   ✅  Collection wiped.")
        except Exception:
            print("   ℹ️   No existing collection to wipe.")

    print(f"🔄  Generating embeddings with {rag_cfg.embedding_model}…")
    print(f"   (This calls the OpenAI API for {len(chunks)} chunks)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=rag_cfg.collection_name,
        persist_directory=chroma_path,
    )

    total = vectorstore._collection.count()
    print(f"   ✅  {total} vectors stored in ChromaDB at '{chroma_path}'")


def main():
    args = parse_args()

    # Validate PDF
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌  PDF not found: {pdf_path}")
        sys.exit(1)

    # Load config
    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        print(f"❌  {e}")
        sys.exit(1)
    rag_cfg = settings.rag

    # Resolve API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("❌  No API key provided.")
        sys.exit(1)

    print(f"\n📄  Ingesting: {pdf_path.name}")
    print(f"    Collection : {rag_cfg.collection_name}")
    print(f"    Chunk size : {rag_cfg.chunk_size} chars (overlap {rag_cfg.chunk_overlap})")
    print(f"    Vector store: {rag_cfg.chroma_path}\n")

    # Step 1: Extract text
    if args.ocr:
        print("🔄  Forced OCR mode…")
        pages = extract_text_ocr(str(pdf_path))
    else:
        pages = extract_text_pypdf(str(pdf_path))
        if not has_text(pages):
            print("⚠️   Most pages appear empty — switching to OCR automatically.")
            pages = extract_text_ocr(str(pdf_path))
        else:
            print(f"✅  Extracted text from {len(pages)} pages via PyPDF.")

    # Step 2: Chunk
    chunks = chunk_documents(pages, rag_cfg.chunk_size, rag_cfg.chunk_overlap)

    # Step 3: Embed + store
    embed_and_store(chunks, settings, api_key, reset=args.reset)

    print("\n✅  Ingestion complete. Run `streamlit run app.py` to start the assistant.\n")


if __name__ == "__main__":
    main()
