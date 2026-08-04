#!/usr/bin/env python3
"""
scripts/documents.py
Inspect and manage the documents in a tenant's collection.

Until now the only lifecycle operation was `ingest.py --reset`, which wipes
everything. This lists what is indexed and removes one document at a time.

Usage:
    python scripts/documents.py list --tenant acme
    python scripts/documents.py delete --tenant acme --doc-id old-manual
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.settings import ConfigError, UnknownTenantError, load_settings

load_dotenv()

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Manage indexed documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tenant", required=True, help="Tenant to operate on")
    common.add_argument("--config", default=None, help="Path to config.yaml")

    sub.add_parser("list", parents=[common], help="List indexed documents")

    delete = sub.add_parser("delete", parents=[common], help="Delete one document")
    delete.add_argument("--doc-id", required=True, help="Document to remove")

    return parser.parse_args()


def open_store(settings, tenant):
    """Open the tenant's collection without loading the reranker or LLM client."""
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌  OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    embeddings = OpenAIEmbeddings(
        model=settings.rag.embedding_model, api_key=api_key
    )
    return Chroma(
        collection_name=tenant.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.rag.chroma_path),
    )


def main() -> int:
    args = parse_args()

    try:
        settings = load_settings(args.config)
        tenant = settings.tenant(args.tenant)
    except (ConfigError, UnknownTenantError) as e:
        print(f"❌  {e}", file=sys.stderr)
        return 1

    from rag.documents import delete_document, list_documents

    store = open_store(settings, tenant)

    if args.command == "list":
        docs = list_documents(store)
        if not docs:
            print(f"No documents indexed for tenant '{tenant.tenant_id}'.")
            return 0

        print(f"\n📚  Documents in '{tenant.tenant_id}' ({tenant.collection_name}):\n")
        print(f"    {'DOC ID':<28} {'CLASSIFICATION':<16} {'CHUNKS':>7}  TITLE")
        for doc in docs:
            print(
                f"    {doc.doc_id:<28} {doc.classification:<16} "
                f"{doc.chunk_count:>7}  {doc.title}"
            )
        print(f"\n    {len(docs)} document(s)\n")
        return 0

    if args.command == "delete":
        removed = delete_document(store, args.doc_id)
        if removed:
            print(f"🗑️   Deleted '{args.doc_id}' ({removed} chunks) from "
                  f"'{tenant.tenant_id}'.")
            return 0
        print(
            f"⚠️   No document '{args.doc_id}' in tenant '{tenant.tenant_id}'.",
            file=sys.stderr,
        )
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
