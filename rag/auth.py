"""
rag/auth.py
Authentication and role lookup.

Credentials live in config/users.yaml so they never sit in source code.
The users file path is injectable, which lets tests run against a fixture
instead of the real credential store.

⚠️  Passwords are still compared in plain text — see config/users.yaml.
    Replace with hashed credentials or an external IdP before production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.settings import Settings, resolve_path, users_path


@dataclass(frozen=True)
class User:
    username: str
    role: str
    display_name: str


def load_users(path: str | os.PathLike | None = None) -> dict:
    """Read the raw users mapping from YAML. Returns {} if the file is absent."""
    resolved = resolve_path(path) if path is not None else users_path()
    if not resolved.exists():
        return {}
    with open(resolved, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("users", {}) or {}


def authenticate(
    username: str,
    password: str,
    path: str | os.PathLike | None = None,
) -> User | None:
    """Validate credentials. Returns a User on success, None on failure."""
    users = load_users(path)
    record = users.get(username)
    if record and record.get("password") == password:
        return User(
            username=username,
            role=record.get("role", ""),
            display_name=record.get("display_name", username),
        )
    return None


def authorize_query(user: User, settings: Settings) -> bool:
    """
    Whether this user's role is permitted to run queries at all.

    config.yaml has always defined `can_query` per role; this is the check
    that actually enforces it.
    """
    return settings.permissions_for(user.role).can_query


def default_users_path() -> Path:
    """Resolved location of the credential file, for display in diagnostics."""
    return users_path()
