"""QueryLog writes where it is told and creates nothing until it does."""

from __future__ import annotations

import json

from rag.logger import QueryLog


def test_directory_not_created_until_first_write(tmp_path):
    target = tmp_path / "logs"
    QueryLog(target)
    assert not target.exists()


def test_log_query_roundtrip(tmp_path):
    log = QueryLog(tmp_path / "logs")
    log.log_query("alice", "admin", "How do I reset it?", "Hold the button.", ["Page 4"])

    entries = log.load_queries()
    assert len(entries) == 1
    assert entries[0]["username"] == "alice"
    assert entries[0]["role"] == "admin"
    assert entries[0]["sources"] == ["Page 4"]
    assert "timestamp" in entries[0]


def test_feedback_truncates_long_answers(tmp_path):
    log = QueryLog(tmp_path / "logs")
    log.save_feedback("guest", "viewer", "q", "x" * 1000, "not_useful")

    entries = log.load_feedback()
    assert len(entries[0]["answer"]) == 300
    assert entries[0]["feedback"] == "not_useful"


def test_reads_are_empty_before_any_write(tmp_path):
    log = QueryLog(tmp_path / "logs")
    assert log.load_queries() == []
    assert log.load_feedback() == []


def test_appends_rather_than_overwrites(tmp_path):
    log = QueryLog(tmp_path / "logs")
    for i in range(3):
        log.log_query("alice", "admin", f"q{i}", "a", [])
    assert [e["question"] for e in log.load_queries()] == ["q0", "q1", "q2"]


def test_queries_and_feedback_use_separate_files(tmp_path):
    log = QueryLog(tmp_path / "logs")
    log.log_query("alice", "admin", "q", "a", [])
    log.save_feedback("alice", "admin", "q", "a", "useful")
    assert len(log.load_queries()) == 1
    assert len(log.load_feedback()) == 1


def test_non_ascii_survives_the_roundtrip(tmp_path):
    log = QueryLog(tmp_path / "logs")
    log.log_query("alice", "admin", "¿Cuál es el par de apriete?", "34 N·m", ["Page 9"])
    assert log.load_queries()[0]["answer"] == "34 N·m"


def test_two_logs_are_isolated(tmp_path):
    a, b = QueryLog(tmp_path / "a"), QueryLog(tmp_path / "b")
    a.log_query("alice", "admin", "only-in-a", "x", [])
    assert b.load_queries() == []


def test_written_lines_are_valid_jsonl(tmp_path):
    log = QueryLog(tmp_path / "logs")
    log.log_query("alice", "admin", "q", "a", [])
    log.log_query("bob", "support", "q2", "a2", [])
    raw = log.query_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(raw) == 2
    assert all(json.loads(line) for line in raw)
