"""Shared fixtures. Every test runs against temp files, never the real config."""

from __future__ import annotations

import pytest
import yaml

SAMPLE_CONFIG = {
    "app": {
        "title": "Test Assistant",
        "icon": "🧪",
        "persona": "a test assistant",
        "description": "Testing.",
    },
    "rag": {
        "collection_prefix": "test_docs",
        "chroma_path": "./chroma_test",
        "chunk_size": 400,
        "chunk_overlap": 80,
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "default_classification": "internal",
    },
    "security": {"classifications": ["public", "internal", "confidential"]},
    "tenants": {
        "acme": {"display_name": "Acme Manufacturing"},
        "globex": {"display_name": "Globex Corporation"},
    },
    "roles": {
        "admin": {
            "can_query": True,
            "can_see_sources": True,
            "max_results": 8,
            "top_n_rerank": 5,
            "clearance": ["public", "internal", "confidential"],
        },
        "support": {
            "can_query": True,
            "can_see_sources": True,
            "max_results": 5,
            "top_n_rerank": 3,
            "clearance": ["public", "internal"],
        },
        "viewer": {
            "can_query": True,
            "can_see_sources": False,
            "max_results": 3,
            "top_n_rerank": 2,
            "clearance": ["public"],
        },
        "suspended": {"can_query": False, "can_see_sources": False},
    },
}

SAMPLE_USERS = {
    "users": {
        "alice": {
            "password": "alice123", "role": "admin",
            "tenant": "acme", "display_name": "Alice",
        },
        "guest": {
            "password": "guest123", "role": "viewer",
            "tenant": "acme", "display_name": "Guest",
        },
        "mallory": {
            "password": "hunter2", "role": "suspended",
            "tenant": "acme", "display_name": "Mal",
        },
        "carol": {
            "password": "carol123", "role": "admin",
            "tenant": "globex", "display_name": "Carol",
        },
        # Deliberately points at a tenant that does not exist in config.
        "orphan": {
            "password": "orphan123", "role": "admin",
            "tenant": "nowhere", "display_name": "Orphan",
        },
    }
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Stop the developer's real environment leaking into assertions."""
    for var in ("RAG_CONFIG", "RAG_USERS", "LOG_DIR", "CHROMA_PATH"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(SAMPLE_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def users_file(tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text(yaml.safe_dump(SAMPLE_USERS), encoding="utf-8")
    return path


@pytest.fixture
def settings(config_file):
    from rag.settings import load_settings

    return load_settings(config_file)


@pytest.fixture
def acme(settings):
    """The default tenant for tests that are not about multi-tenancy."""
    return settings.tenant("acme")


@pytest.fixture
def globex(settings):
    return settings.tenant("globex")
