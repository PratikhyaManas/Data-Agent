import sys
from pathlib import Path

from project_bootstrap import ensure_repo_root


def test_ensure_repo_root_adds_repo_root_to_path():
    repo_root = str(Path(__file__).resolve().parents[1])
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != Path(repo_root)]

    ensure_repo_root()

    assert repo_root in sys.path
