"""Authentication against an injected credential file."""

from __future__ import annotations

from rag.auth import authenticate, authorize_query, load_users


def test_valid_credentials_return_user(users_file):
    user = authenticate("alice", "alice123", users_file)
    assert user is not None
    assert user.username == "alice"
    assert user.role == "admin"
    assert user.display_name == "Alice"


def test_wrong_password_returns_none(users_file):
    assert authenticate("alice", "wrong", users_file) is None


def test_unknown_user_returns_none(users_file):
    assert authenticate("nobody", "alice123", users_file) is None


def test_empty_password_does_not_authenticate(users_file):
    assert authenticate("alice", "", users_file) is None


def test_missing_users_file_denies_everyone(tmp_path):
    assert load_users(tmp_path / "absent.yaml") == {}
    assert authenticate("alice", "alice123", tmp_path / "absent.yaml") is None


def test_authorize_query_allows_permitted_role(users_file, settings):
    user = authenticate("alice", "alice123", users_file)
    assert authorize_query(user, settings) is True


def test_authorize_query_blocks_role_with_can_query_false(users_file, settings):
    """This is the check config.yaml always described but nothing enforced."""
    user = authenticate("mallory", "hunter2", users_file)
    assert user.role == "suspended"
    assert authorize_query(user, settings) is False


def test_authorize_query_blocks_role_missing_from_config(users_file, settings):
    from rag.auth import User

    assert authorize_query(User("x", "ghost-role", "X"), settings) is False
