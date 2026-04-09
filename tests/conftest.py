"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemon.config import MnemonConfig


@pytest.fixture
def mnemon_dir(tmp_path: Path) -> Path:
    """Create a temporary mnemon directory structure."""
    base = tmp_path / "mnemon"
    base.mkdir()
    (base / "canonical").mkdir()
    (base / "chroma").mkdir()
    return base


@pytest.fixture
def config(mnemon_dir: Path) -> MnemonConfig:
    """Create a MnemonConfig pointing at the temp directory."""
    return MnemonConfig(base_dir=mnemon_dir)
