"""
The point of this suite: rag/ must never import a UI framework again.

The check runs in a subprocess because sys.modules is process-global — if any
other test had already pulled in Streamlit, an in-process assertion would be
meaningless.
"""

from __future__ import annotations

import subprocess
import sys

from rag.settings import PROJECT_ROOT

PROBE = """
import sys
import rag
import rag.auth, rag.logger, rag.settings, rag.retrieval, rag.generation

leaked = [m for m in sys.modules if m == "streamlit" or m.startswith("streamlit.")]
if leaked:
    raise SystemExit("streamlit imported by rag/: " + ", ".join(sorted(leaked)[:5]))
print("clean")
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_rag_package_does_not_import_streamlit():
    result = _run(PROBE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_rag_imports_from_an_unrelated_working_directory(tmp_path):
    """Paths resolve off the package, not the CWD."""
    code = (
        "from rag.settings import config_path, users_path, PROJECT_ROOT\n"
        "assert config_path().is_absolute()\n"
        "assert users_path().is_absolute()\n"
        "print(PROJECT_ROOT)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,  # deliberately not the project root
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(PROJECT_ROOT)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(PROJECT_ROOT) in result.stdout


def test_importing_rag_creates_no_directories(tmp_path):
    """rag/logger.py used to mkdir at import time."""
    code = (
        "import rag, rag.logger\n"
        "import os\n"
        "print(sorted(os.listdir('.')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(PROJECT_ROOT)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "[]"
