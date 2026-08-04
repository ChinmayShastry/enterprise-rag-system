"""
rag — framework-free retrieval-augmented generation package.

Nothing in this package imports Streamlit. The UI layer (app.py) is one
consumer; the CLI (scripts/query.py) and the test suite are others.

Only the lightweight modules are re-exported here. `rag.retrieval` and
`rag.generation` pull in heavier dependencies, so import them directly:

    from rag.retrieval import build_retriever
    from rag.generation import generate_answer
"""

from rag.auth import User, authenticate, authorize_query, load_users
from rag.logger import QueryLog
from rag.settings import (
    PROJECT_ROOT,
    ConfigError,
    RolePermissions,
    Settings,
    get_settings,
    load_settings,
)

__all__ = [
    "PROJECT_ROOT",
    "ConfigError",
    "QueryLog",
    "RolePermissions",
    "Settings",
    "User",
    "authenticate",
    "authorize_query",
    "get_settings",
    "load_settings",
    "load_users",
]
