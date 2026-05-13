"""
rag/logger.py
Query and feedback logging to JSONL files.
The logs directory is gitignored — logs are runtime output, not source code.
"""

import json
import os
import datetime
from pathlib import Path

_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = _LOG_DIR / "query_log.jsonl"
FEEDBACK_FILE = _LOG_DIR / "feedback_log.jsonl"


def log_query(user: dict, question: str, answer: str, sources: list[str]) -> None:
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": user["username"],
        "role": user["role"],
        "question": question,
        "answer": answer,
        "sources": sources,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_feedback(user: dict, question: str, answer: str, feedback: str) -> None:
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": user["username"],
        "role": user["role"],
        "question": question,
        "answer": answer[:300],
        "feedback": feedback,  # "useful" | "not_useful"
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_logs() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
