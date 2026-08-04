#!/usr/bin/env python3
"""
scripts/query.py
Answer a question from the command line — no Streamlit, no browser.

This exists to keep rag/ honest: if retrieval or generation ever imports a UI
framework again, this script breaks immediately.

Usage:
    python scripts/query.py "How do I fix error code E4?"
    python scripts/query.py "What is the notice period?" --role support
    python scripts/query.py "Max load?" --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag.access import AccessPolicy
from rag.generation import generate_answer, get_sources
from rag.settings import ConfigError, load_settings

load_dotenv()

# Windows consoles default to cp1252 and cannot encode this script's emoji
# output; without this a status line raises UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Query the RAG system from the CLI.")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--role",
        default="admin",
        help="Role whose retrieval depth and permissions to apply (default: admin)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON instead of text"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        print(f"❌  {e}", file=sys.stderr)
        return 1

    permissions = settings.permissions_for(args.role)
    if not permissions.can_query:
        print(
            f"❌  Role '{args.role}' is not permitted to query "
            f"(can_query is false or the role is undefined).",
            file=sys.stderr,
        )
        return 2

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌  OPENAI_API_KEY is not set. Add it to .env or the environment.",
              file=sys.stderr)
        return 1

    # Imported here so --help and the guard clauses above stay fast.
    from rag.retrieval import build_retriever

    retriever = build_retriever(settings, api_key)
    if retriever.chunk_count == 0:
        print("❌  No documents indexed. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    policy = AccessPolicy.from_permissions(args.role, permissions)
    docs = retriever.retrieve(
        args.question,
        max_results=permissions.max_results,
        top_n=permissions.top_n_rerank,
        policy=policy,
    )

    if not docs:
        clearance = ", ".join(sorted(policy.clearance)) or "none"
        print(
            f"No documents matching the '{args.role}' clearance ({clearance}) "
            f"contain an answer to that question.",
            file=sys.stderr,
        )
        return 3

    answer = generate_answer(
        retriever.client,
        args.question,
        docs,
        llm_model=settings.rag.llm_model,
        persona=settings.app.persona,
    )
    sources = get_sources(docs) if permissions.can_see_sources else []

    if args.as_json:
        print(json.dumps(
            {
                "question": args.question,
                "role": args.role,
                "answer": answer,
                "sources": sources,
                "chunks_used": len(docs),
            },
            indent=2,
            ensure_ascii=False,
        ))
    else:
        print(f"\n{answer}\n")
        if sources:
            print(f"📄 Sources: {', '.join(sources)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
