"""Paths the tests read: the repository root and the pinned fixture data."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
"""The repository root, whichever tier directory a test lives in."""

CONTRACT_DATA = REPO_ROOT / "tests" / "integration" / "contracts" / "data"
"""Outputs of other stack libraries, pinned at a commit (see each file's test)."""
