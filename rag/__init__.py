"""
rag — framework-free retrieval-augmented generation package.

Nothing in this package imports Streamlit. The UI layer (app.py) is one
consumer; the CLI (scripts/query.py) and the test suite are others.

Only the lightweight modules are re-exported here. `rag.retrieval` and
`rag.generation` pull in heavier dependencies, so import them directly:

    from rag.retrieval import build_retriever
    from rag.generation import generate_answer
"""

from rag.access import AccessPolicy
from rag.auth import (
    User,
    authenticate,
    authorize_query,
    authorize_tenant,
    load_users,
)
from rag.documents import DocumentInfo, chunk_id, doc_id_from_path, slugify
from rag.logger import QueryLog
from rag.settings import (
    PROJECT_ROOT,
    ConfigError,
    RolePermissions,
    Settings,
    TenantConfig,
    UnknownTenantError,
    get_settings,
    load_settings,
)

__all__ = [
    "PROJECT_ROOT",
    "AccessPolicy",
    "ConfigError",
    "DocumentInfo",
    "QueryLog",
    "RolePermissions",
    "Settings",
    "TenantConfig",
    "UnknownTenantError",
    "User",
    "authenticate",
    "authorize_query",
    "authorize_tenant",
    "chunk_id",
    "doc_id_from_path",
    "get_settings",
    "load_settings",
    "load_users",
    "slugify",
]
