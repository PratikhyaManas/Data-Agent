from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root() -> str:
    """Ensure the repository root is present on sys.path for local imports."""
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root
