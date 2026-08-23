#!/usr/bin/env python3
"""
scripts/ingest.py
Ingest a PDF into a tenant's collection from the command line.

The pipeline itself lives in rag/ingestion.py; this file only parses arguments
and prints progress, so the CLI and the admin upload page in app.py cannot
drift apart.

Usage:
    python scripts/ingest.py --tenant demo --pdf data/manual.pdf
    python scripts/ingest.py --tenant demo --pdf data/scanned.pdf --ocr
    python scripts/ingest.py --tenant demo --pdf data/v2.pdf --doc-id manual
    python scripts/ingest.py --tenant demo --pdf data/manual.pdf --reset
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.ingestion import IngestionError, ingest_document, roles_cleared_for
from rag.settings import ConfigError, UnknownTenantError, load_settings

load_dotenv()

# Windows consoles default to cp1252, which cannot encode the emoji in this
# script's output — without this, a status line raises UnicodeEncodeError and
# masks the actual result.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument(
        "--tenant",
        required=True,
        help="Tenant that will own this document. Required — data placement is "
             "never implicit.",
    )
    parser.add_argument("--ocr", action="store_true", help="Force OCR extraction")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe the tenant's collection first"
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Stable document identifier (default: slug of the filename). "
             "Re-ingesting the same doc-id replaces that document.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Human-readable document name (default: the filename)",
    )
    parser.add_argument(
        "--classification",
        default=None,
        help="Security label for this document (default: rag.default_classification). "
             "Roles retrieve only the classifications they are cleared for.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = load_settings(args.config)
        tenant = settings.tenant(args.tenant)
    except (ConfigError, UnknownTenantError) as e:
        print(f"❌  {e}")
        return 1

    classification = args.classification or settings.rag.default_classification
    cleared = roles_cleared_for(settings, classification)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("❌  No API key provided.")
        return 1

    print(f"\n📄  Ingesting: {Path(args.pdf).name}")
    print(f"    Tenant     : {tenant.tenant_id} ({tenant.display_name})")
    print(f"    Collection : {tenant.collection_name}")
    print(f"    Chunk size : {settings.rag.chunk_size} chars "
          f"(overlap {settings.rag.chunk_overlap})")
    print(f"    Vector store  : {settings.rag.chroma_path}")
    print(f"    Classification: {classification}")
    print(f"    Retrievable by: {', '.join(cleared) if cleared else '⚠️  no role'}\n")

    try:
        result = ingest_document(
            args.pdf,
            settings=settings,
            tenant_id=args.tenant,
            api_key=api_key,
            doc_id=args.doc_id,
            title=args.title,
            classification=args.classification,
            force_ocr=args.ocr,
            reset=args.reset,
            on_progress=lambda message: print(f"   •  {message}"),
        )
    except (IngestionError, UnknownTenantError) as e:
        print(f"❌  {e}")
        return 1

    print(f"\n✅  Stored '{result.doc_id}' — {result.chunk_count} chunks "
          f"from {result.page_count} pages.")
    if result.replaced:
        print(f"    Replaced {result.replaced_chunks} chunks of the previous version.")
    print(f"    Collection '{tenant.collection_name}' now holds "
          f"{result.collection_total} vectors.")
    print("\n    Run `streamlit run app.py` to start the assistant.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
