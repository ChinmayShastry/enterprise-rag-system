"""
rag/settings.py
Configuration loading, path resolution, and typed settings.

This module is deliberately framework-free — it knows nothing about Streamlit,
so the same settings object drives the web app, the ingestion script, the
headless CLI, and the test suite.

Paths are resolved relative to the package, not the process working directory,
so the system behaves identically under `streamlit run`, `python -m pytest`,
Docker, and cron.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# ─────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────

# Anchored to this file rather than os.getcwd(), so imports work no matter
# where the interpreter was started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_USERS_PATH = PROJECT_ROOT / "config" / "users.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def resolve_path(value: str | os.PathLike, base: Path = PROJECT_ROOT) -> Path:
    """
    Turn a possibly-relative config value into an absolute path.
    Relative paths resolve against the project root, never the CWD.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def config_path() -> Path:
    """Config file location, overridable with the RAG_CONFIG env var."""
    override = os.getenv("RAG_CONFIG")
    return resolve_path(override) if override else DEFAULT_CONFIG_PATH


def users_path() -> Path:
    """User credential file location, overridable with the RAG_USERS env var."""
    override = os.getenv("RAG_USERS")
    return resolve_path(override) if override else DEFAULT_USERS_PATH


def log_dir() -> Path:
    """Log directory, overridable with the LOG_DIR env var."""
    override = os.getenv("LOG_DIR")
    return resolve_path(override) if override else DEFAULT_LOG_DIR


# ─────────────────────────────────────────────────────────────────
# Typed configuration
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    title: str
    icon: str
    persona: str
    description: str


# Used when config.yaml predates the security section.
DEFAULT_CLASSIFICATIONS: tuple[str, ...] = ("public", "internal", "confidential")


@dataclass(frozen=True)
class RagConfig:
    collection_name: str
    chroma_path: Path
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    llm_model: str
    reranker_model: str
    default_classification: str


@dataclass(frozen=True)
class RolePermissions:
    can_query: bool
    can_see_sources: bool
    max_results: int
    top_n_rerank: int
    # The real access boundary: which document classifications this role may
    # retrieve. An empty set means the role can retrieve nothing.
    clearance: frozenset[str]


# Deny-by-default. Any role missing from config.yaml gets this rather than
# inheriting something permissive.
RESTRICTED = RolePermissions(
    can_query=False,
    can_see_sources=False,
    max_results=2,
    top_n_rerank=1,
    clearance=frozenset(),
)


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    rag: RagConfig
    roles: dict[str, RolePermissions]
    classifications: tuple[str, ...]
    source_path: Path

    def permissions_for(self, role: str) -> RolePermissions:
        """Permissions for a role, falling back to RESTRICTED if undefined."""
        return self.roles.get(role, RESTRICTED)


# ─────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing or structurally invalid."""


def _require(section: dict, key: str, where: str):
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return section[key]


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """
    Read and validate config.yaml into a typed Settings object.

    Fails loudly at load time on a malformed config rather than raising a
    KeyError deep inside a request.
    """
    resolved = resolve_path(path) if path is not None else config_path()
    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    with open(resolved, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    app_raw = _require(raw, "app", resolved.name)
    rag_raw = _require(raw, "rag", resolved.name)
    roles_raw = raw.get("roles", {})

    app = AppConfig(
        title=app_raw.get("title", "Enterprise RAG Assistant"),
        icon=app_raw.get("icon", "🤖"),
        persona=app_raw.get("persona", "a helpful assistant for company documentation"),
        description=app_raw.get("description", "Ask questions about your documents."),
    )

    # CHROMA_PATH env var wins over config, so deployments can point at a
    # mounted volume without editing the file.
    chroma_raw = os.getenv("CHROMA_PATH") or rag_raw.get("chroma_path", "./chroma_db")

    classifications = tuple(
        (raw.get("security") or {}).get("classifications") or DEFAULT_CLASSIFICATIONS
    )

    default_classification = rag_raw.get("default_classification", classifications[-1])
    if default_classification not in classifications:
        raise ConfigError(
            f"rag.default_classification '{default_classification}' is not one of "
            f"the declared classifications {list(classifications)}"
        )

    rag = RagConfig(
        collection_name=_require(rag_raw, "collection_name", "rag"),
        chroma_path=resolve_path(chroma_raw),
        chunk_size=int(rag_raw.get("chunk_size", 500)),
        chunk_overlap=int(rag_raw.get("chunk_overlap", 100)),
        embedding_model=_require(rag_raw, "embedding_model", "rag"),
        llm_model=_require(rag_raw, "llm_model", "rag"),
        reranker_model=_require(rag_raw, "reranker_model", "rag"),
        default_classification=default_classification,
    )

    roles = {
        name: RolePermissions(
            can_query=bool(spec.get("can_query", False)),
            can_see_sources=bool(spec.get("can_see_sources", False)),
            max_results=int(spec.get("max_results", RESTRICTED.max_results)),
            top_n_rerank=int(spec.get("top_n_rerank", RESTRICTED.top_n_rerank)),
            clearance=frozenset(spec.get("clearance") or ()),
        )
        for name, spec in (roles_raw or {}).items()
    }

    _validate_clearances(roles, classifications)

    return Settings(
        app=app,
        rag=rag,
        roles=roles,
        classifications=classifications,
        source_path=resolved,
    )


def _validate_clearances(
    roles: dict[str, RolePermissions],
    classifications: tuple[str, ...],
) -> None:
    """
    Catch access-control typos at startup rather than in production.

    A misspelt clearance entry would otherwise silently grant nothing, and a
    queryable role with no clearance would silently return zero results — both
    look like retrieval bugs rather than config mistakes.
    """
    known = set(classifications)
    for name, perms in roles.items():
        unknown = perms.clearance - known
        if unknown:
            raise ConfigError(
                f"Role '{name}' is cleared for unknown classification(s) "
                f"{sorted(unknown)}; declared classifications are {list(classifications)}"
            )
        if perms.can_query and not perms.clearance:
            raise ConfigError(
                f"Role '{name}' has can_query: true but an empty clearance list, "
                f"so it could never retrieve anything. Grant it a clearance or "
                f"set can_query: false."
            )


@lru_cache(maxsize=8)
def get_settings(path: str | os.PathLike | None = None) -> Settings:
    """
    Cached variant of load_settings() for callers that read config often.
    Call get_settings.cache_clear() after editing config.yaml.
    """
    return load_settings(path)
