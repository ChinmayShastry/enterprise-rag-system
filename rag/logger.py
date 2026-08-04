"""
rag/logger.py
Query and feedback logging to JSONL files.

Previously this module created its log directory at import time and kept the
file paths in module-level globals. That made importing the package a
filesystem side effect and left tests writing into the real log directory.
QueryLog now takes its directory as a constructor argument and creates it
lazily on first write.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from rag.settings import log_dir, resolve_path


def _now() -> str:
    return datetime.datetime.now().isoformat()


class QueryLog:
    """
    Append-only JSONL log of queries and feedback.

    Pass `directory` to isolate a test run, a tenant, or a deployment;
    omit it to use the LOG_DIR env var or the project's logs/ folder.
    """

    def __init__(self, directory: str | os.PathLike | None = None):
        self.directory = resolve_path(directory) if directory is not None else log_dir()
        self.query_file = self.directory / "query_log.jsonl"
        self.feedback_file = self.directory / "feedback_log.jsonl"

    # ── writes ───────────────────────────────────────────────────

    def _append(self, path: Path, entry: dict) -> None:
        # Created on first write, not on import.
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_query(
        self,
        username: str,
        role: str,
        question: str,
        answer: str,
        sources: list[str],
    ) -> None:
        self._append(
            self.query_file,
            {
                "timestamp": _now(),
                "username": username,
                "role": role,
                "question": question,
                "answer": answer,
                "sources": sources,
            },
        )

    def save_feedback(
        self,
        username: str,
        role: str,
        question: str,
        answer: str,
        feedback: str,
    ) -> None:
        """`feedback` is "useful" or "not_useful"."""
        self._append(
            self.feedback_file,
            {
                "timestamp": _now(),
                "username": username,
                "role": role,
                "question": question,
                "answer": answer[:300],
                "feedback": feedback,
            },
        )

    # ── reads ────────────────────────────────────────────────────

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def load_queries(self) -> list[dict]:
        return self._read(self.query_file)

    def load_feedback(self) -> list[dict]:
        return self._read(self.feedback_file)
