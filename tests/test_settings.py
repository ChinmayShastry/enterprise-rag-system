"""Config loading, path resolution, and deny-by-default permissions."""

from __future__ import annotations

import pytest

from rag.settings import (
    RESTRICTED,
    ConfigError,
    PROJECT_ROOT,
    config_path,
    load_settings,
    log_dir,
    resolve_path,
    users_path,
)


def test_loads_typed_sections(settings):
    assert settings.app.title == "Test Assistant"
    assert settings.rag.collection_prefix == "test_docs"
    assert settings.rag.chunk_size == 400
    assert settings.rag.chunk_overlap == 80


def test_relative_chroma_path_resolves_against_project_root(settings):
    assert settings.rag.chroma_path.is_absolute()
    assert settings.rag.chroma_path.name == "chroma_test"


def test_chroma_path_env_var_wins(config_file, monkeypatch, tmp_path):
    override = tmp_path / "mounted_volume"
    monkeypatch.setenv("CHROMA_PATH", str(override))
    assert load_settings(config_file).rag.chroma_path == override.resolve()


def test_missing_config_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


def test_structurally_invalid_config_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("app: {title: x}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="rag"):
        load_settings(bad)


def test_missing_required_rag_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "app: {}\nrag: {collection_prefix: c, embedding_model: e, llm_model: l}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="reranker_model"):
        load_settings(bad)


def test_legacy_collection_name_raises_a_migration_error(tmp_path, config_file):
    """
    Reusing the old single-tenant key as a prefix would silently repoint every
    query at a different, empty collection. Refuse instead.
    """
    import yaml

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["rag"]["collection_name"] = raw["rag"].pop("collection_prefix")
    bad = tmp_path / "legacy.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="collection_prefix"):
        load_settings(bad)


def test_known_role_permissions(settings):
    admin = settings.permissions_for("admin")
    assert admin.can_query is True
    assert admin.top_n_rerank == 5

    viewer = settings.permissions_for("viewer")
    assert viewer.can_see_sources is False
    assert viewer.max_results == 3


def test_unknown_role_gets_restricted_defaults(settings):
    assert settings.permissions_for("does-not-exist") == RESTRICTED
    assert settings.permissions_for("does-not-exist").can_query is False


def test_role_omitting_can_query_defaults_to_denied(settings):
    """A role defined without can_query must not silently become permitted."""
    assert settings.permissions_for("suspended").can_query is False


def test_default_paths_are_absolute_and_under_project_root():
    for path in (config_path(), users_path(), log_dir()):
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents or path.parent == PROJECT_ROOT


def test_env_overrides_for_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CONFIG", str(tmp_path / "c.yaml"))
    monkeypatch.setenv("RAG_USERS", str(tmp_path / "u.yaml"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    assert config_path() == (tmp_path / "c.yaml").resolve()
    assert users_path() == (tmp_path / "u.yaml").resolve()
    assert log_dir() == (tmp_path / "l").resolve()


def test_resolve_path_leaves_absolute_paths_alone(tmp_path):
    assert resolve_path(tmp_path) == tmp_path


def test_editing_config_is_visible_to_a_fresh_load(config_file):
    """
    Regression: config.yaml was cached for the life of the process while
    users.yaml was re-read on every login. Renaming a tenant in both left a
    running app matching new users against old tenants, reported as "account X
    is assigned to tenant Y, which is not configured".

    load_settings() must therefore always reflect the file on disk; app.py
    caches it keyed on the file's mtime rather than forever.
    """
    import yaml

    assert "demo-tenant" not in load_settings(config_file).tenants

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["tenants"]["demo-tenant"] = {"display_name": "Added Later"}
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert "demo-tenant" in load_settings(config_file).tenants


def test_app_keys_its_settings_cache_on_the_config_file(tmp_path):
    """
    app.py cannot be imported under test, so this reads the source: the
    settings cache must be keyed on something that changes when config.yaml
    changes, not cached unconditionally for the process lifetime.
    """
    import ast

    tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))
    loader = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_settings_cached"
    )
    body = ast.dump(loader)
    assert "st_mtime" in body, (
        "load_settings_cached must key its cache on the config file's "
        "modification time, or edits to config.yaml will not be picked up "
        "while users.yaml is re-read on every login"
    )
