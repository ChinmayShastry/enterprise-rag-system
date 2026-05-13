"""
rag/auth.py
Authentication and role-based access control.
Reads credentials from config/users.yaml so they never live in source code.
"""

import yaml
from pathlib import Path

_USERS_PATH = Path("config/users.yaml")


def _load_users() -> dict:
    with open(_USERS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["users"]


def authenticate(username: str, password: str) -> dict | None:
    """
    Validate credentials. Returns a user dict on success, None on failure.
    """
    users = _load_users()
    user = users.get(username)
    if user and user["password"] == password:
        return {
            "username": username,
            "role": user["role"],
            "display_name": user.get("display_name", username),
        }
    return None


def get_permissions(role: str, config: dict) -> dict:
    """
    Return the permission dict for a given role from config.yaml.
    Falls back to most restrictive defaults if role not found.
    """
    return config["roles"].get(
        role,
        {"can_query": False, "can_see_sources": False, "max_results": 2, "top_n_rerank": 1},
    )
