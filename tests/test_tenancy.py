"""
Tenant isolation.

Isolation here is structural, not filter-based: each tenant's retriever holds
its own Chroma collection, so another tenant's vectors are not present to leak.
The tests below cover the config that establishes that, and the guard that
catches the one realistic way it could be subverted — a retriever cached across
tenants.
"""

from __future__ import annotations

import pytest
import yaml
from langchain_core.documents import Document

from rag.access import AccessPolicy
from rag.auth import authenticate, authorize_tenant
from rag.logger import QueryLog
from rag.retrieval import Retriever, TenantMismatchError
from rag.settings import ConfigError, UnknownTenantError, load_settings
from tests.test_retrieval import (
    REWRITES,
    FakeBM25,
    FakeClient,
    FakeReranker,
    FakeVectorStore,
)


def rewrite(config_file, tmp_path, mutate, name="mutated.yaml"):
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ── tenant configuration ─────────────────────────────────────────


def test_each_tenant_gets_its_own_collection(settings):
    assert settings.tenant("acme").collection_name == "test_docs_acme"
    assert settings.tenant("globex").collection_name == "test_docs_globex"


def test_collections_are_distinct_across_tenants(settings):
    names = {t.collection_name for t in settings.tenants.values()}
    assert len(names) == len(settings.tenants)


def test_explicit_collection_override_is_honoured(config_file, tmp_path):
    def mutate(raw):
        raw["tenants"]["acme"]["collection"] = "acme-legacy-store"

    settings = load_settings(rewrite(config_file, tmp_path, mutate))
    assert settings.tenant("acme").collection_name == "acme-legacy-store"


def test_two_tenants_sharing_a_collection_is_rejected(config_file, tmp_path):
    """The failure that would silently merge two customers' corpora."""

    def mutate(raw):
        raw["tenants"]["acme"]["collection"] = "shared"
        raw["tenants"]["globex"]["collection"] = "shared"

    with pytest.raises(ConfigError, match="must not share a collection"):
        load_settings(rewrite(config_file, tmp_path, mutate))


def test_illegal_collection_name_is_rejected(config_file, tmp_path):
    def mutate(raw):
        raw["tenants"]["acme"]["collection"] = "no"  # too short for Chroma

    with pytest.raises(ConfigError, match="not a valid ChromaDB collection name"):
        load_settings(rewrite(config_file, tmp_path, mutate))


def test_config_without_tenants_is_rejected(config_file, tmp_path):
    def mutate(raw):
        raw.pop("tenants")

    with pytest.raises(ConfigError, match="No tenants configured"):
        load_settings(rewrite(config_file, tmp_path, mutate))


def test_unknown_tenant_raises_rather_than_defaulting(settings):
    """There is no safe tenant to fall back to, so guessing is not an option."""
    with pytest.raises(UnknownTenantError):
        settings.tenant("does-not-exist")


# ── users bound to tenants ───────────────────────────────────────


def test_user_carries_its_tenant(users_file):
    assert authenticate("alice", "alice123", users_file).tenant == "acme"
    assert authenticate("carol", "carol123", users_file).tenant == "globex"


def test_user_with_unconfigured_tenant_is_refused(users_file, settings):
    orphan = authenticate("orphan", "orphan123", users_file)
    assert orphan.tenant == "nowhere"
    assert authorize_tenant(orphan, settings) is False


def test_user_with_a_configured_tenant_is_allowed(users_file, settings):
    assert authorize_tenant(authenticate("alice", "alice123", users_file), settings)


def test_policy_for_user_takes_both_role_and_tenant(users_file, settings):
    policy = AccessPolicy.for_user(authenticate("carol", "carol123", users_file), settings)
    assert policy.tenant == "globex"
    assert policy.role == "admin"


# ── the cross-tenant guard ───────────────────────────────────────

ACME_DOCS = [
    Document(
        page_content="acme torque spec is 34 newton metres",
        metadata={"page": 1, "classification": "public", "tenant": "acme"},
    )
]


def acme_retriever(settings):
    return Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore(ACME_DOCS),
        chunks=ACME_DOCS,
        bm25=FakeBM25([9.9]),
        reranker=FakeReranker(),
        settings=settings,
        tenant=settings.tenant("acme"),
    )


def test_retriever_refuses_a_policy_from_another_tenant(settings):
    """
    Simulates the realistic failure: a cached Acme retriever handed to a Globex
    session. Without this the request would quietly answer Carol from Acme's
    documents.
    """
    retriever = acme_retriever(settings)
    globex_admin = AccessPolicy.for_role("admin", "globex", settings)

    with pytest.raises(TenantMismatchError, match="acme"):
        retriever.retrieve("torque", max_results=5, top_n=3, policy=globex_admin)


def test_search_also_refuses_a_foreign_policy(settings):
    retriever = acme_retriever(settings)
    globex_admin = AccessPolicy.for_role("admin", "globex", settings)

    with pytest.raises(TenantMismatchError):
        retriever.search("torque", k=5, policy=globex_admin)


def test_matching_tenant_is_served_normally(settings):
    retriever = acme_retriever(settings)
    acme_admin = AccessPolicy.for_role("admin", "acme", settings)
    results = retriever.retrieve("torque", max_results=5, top_n=3, policy=acme_admin)
    assert any("34 newton metres" in d.page_content for d in results)


def test_globex_admin_sees_nothing_of_acme(settings):
    """
    A Globex retriever is built from the Globex collection, which simply does
    not contain Acme's chunks — clearance never even comes into it.
    """
    globex = Retriever(
        client=FakeClient(REWRITES),
        vectorstore=FakeVectorStore([]),   # globex collection is empty
        chunks=[],
        bm25=FakeBM25([]),
        reranker=FakeReranker(),
        settings=settings,
        tenant=settings.tenant("globex"),
    )
    policy = AccessPolicy.for_role("admin", "globex", settings)
    assert globex.retrieve("torque", max_results=5, top_n=3, policy=policy) == []


# ── the factory opens the right collection ───────────────────────
#
# Everything above injects a vector store directly, which cannot catch
# build_retriever() opening the wrong collection — the single most important
# step in tenant isolation. These tests stub the heavy imports so the factory
# itself can be exercised without Chroma or torch installed.


def install_vector_stubs(monkeypatch) -> dict:
    """Replace build_retriever's heavy imports; record what Chroma was given."""
    import sys
    import types

    recorded: dict = {}

    class StubChroma:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def get(self):
            return {"documents": [], "metadatas": []}

    def module(name: str, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("langchain_community")
    module("langchain_community.vectorstores", Chroma=StubChroma)
    module("langchain_openai", OpenAIEmbeddings=lambda **kw: ("embeddings", kw))
    module("rank_bm25", BM25Okapi=lambda tokenized: ("bm25", tokenized))
    module("sentence_transformers", CrossEncoder=lambda name: ("reranker", name))
    return recorded


def test_build_retriever_opens_the_tenants_own_collection(settings, monkeypatch):
    from rag.retrieval import build_retriever

    recorded = install_vector_stubs(monkeypatch)
    retriever = build_retriever(settings, "sk-test", "globex")

    assert recorded["collection_name"] == "test_docs_globex"
    assert retriever.tenant.tenant_id == "globex"


def test_build_retriever_gives_each_tenant_a_different_collection(settings, monkeypatch):
    from rag.retrieval import build_retriever

    recorded = install_vector_stubs(monkeypatch)

    build_retriever(settings, "sk-test", "acme")
    acme_collection = recorded["collection_name"]
    build_retriever(settings, "sk-test", "globex")

    assert acme_collection == "test_docs_acme"
    assert recorded["collection_name"] == "test_docs_globex"
    assert acme_collection != recorded["collection_name"]


def test_build_retriever_rejects_an_unknown_tenant(settings, monkeypatch):
    from rag.retrieval import build_retriever

    install_vector_stubs(monkeypatch)
    with pytest.raises(UnknownTenantError):
        build_retriever(settings, "sk-test", "not-a-tenant")


def test_build_retriever_handles_a_tenant_with_no_documents(settings, monkeypatch):
    """
    Regression: a newly onboarded tenant has an empty collection, and the real
    BM25Okapi divides by vocabulary size when computing average idf. Building
    it from an empty corpus raised ZeroDivisionError before the caller ever
    reached the chunk_count check, so signing in as the first user of a new
    tenant crashed instead of showing "no documents indexed".
    """
    import sys

    from rag.retrieval import build_retriever

    install_vector_stubs(monkeypatch)
    # Use the genuine BM25Okapi here — a stub would not reproduce the crash.
    real_bm25 = pytest.importorskip("rank_bm25")
    monkeypatch.setitem(sys.modules, "rank_bm25", real_bm25)

    retriever = build_retriever(settings, "sk-test", "globex")

    assert retriever.chunk_count == 0
    assert retriever.bm25 is None


def test_empty_tenant_retrieves_nothing_rather_than_raising(settings, monkeypatch):
    import sys

    from rag.access import AccessPolicy
    from rag.retrieval import build_retriever

    install_vector_stubs(monkeypatch)
    monkeypatch.setitem(sys.modules, "rank_bm25", pytest.importorskip("rank_bm25"))

    retriever = build_retriever(settings, "sk-test", "globex")
    policy = AccessPolicy.for_role("admin", "globex", settings)

    assert retriever.retrieve("anything", max_results=5, top_n=3, policy=policy) == []


# ── the UI cache key ─────────────────────────────────────────────


def test_app_caches_the_retriever_per_tenant():
    """
    app.py cannot be imported under test — it is a Streamlit script that runs
    top to bottom. But its cache key is a genuine cross-tenant leak vector: a
    @st.cache_resource keyed on the API key alone hands one tenant's retriever
    to every tenant sharing that key. So this reads the source instead.
    """
    import ast

    from rag.settings import PROJECT_ROOT

    tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    loader = functions.get("load_retriever")
    assert loader is not None, "app.py no longer defines load_retriever"

    params = [arg.arg for arg in loader.args.args]
    assert "tenant_id" in params, (
        "load_retriever must take tenant_id so it is part of the "
        f"@st.cache_resource key; got {params}"
    )

    decorators = ast.dump(ast.Module(body=loader.decorator_list, type_ignores=[]))
    assert "cache_resource" in decorators


def test_app_scopes_the_query_log_per_tenant():
    import ast

    from rag.settings import PROJECT_ROOT

    tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))
    loader = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_query_log"
        ),
        None,
    )
    assert loader is not None, "app.py no longer defines load_query_log"
    assert "tenant_id" in [arg.arg for arg in loader.args.args], (
        "load_query_log must be scoped per tenant, or one tenant's admin "
        "dashboard would show another tenant's queries"
    )


# ── audit trail isolation ────────────────────────────────────────


def test_tenant_logs_are_separate_files(tmp_path):
    acme = QueryLog.for_tenant("acme", tmp_path)
    globex = QueryLog.for_tenant("globex", tmp_path)

    acme.log_query("alice", "admin", "acme question", "a", [], tenant="acme")

    assert len(acme.load_queries()) == 1
    assert globex.load_queries() == []


def test_logged_entries_record_the_tenant(tmp_path):
    log = QueryLog.for_tenant("acme", tmp_path)
    log.log_query("alice", "admin", "q", "a", [], tenant="acme")
    assert log.load_queries()[0]["tenant"] == "acme"
