"""Tests for backup preparation and release."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemon.api import Memory


@pytest.fixture
def mem(tmp_path: Path):
    """Create a Memory instance backed by a temp directory."""
    m = Memory(base_dir=tmp_path / "mnemon")
    yield m
    m.close()


class TestBackupPrep:
    def test_backup_prep_creates_lock(self, mem: Memory):
        lock_path = mem.backup_prep()
        assert Path(lock_path).exists()
        # Clean up so fixture close works after reopen
        mem.backup_release()

    def test_ingest_blocked_during_backup(self, mem: Memory):
        mem.backup_prep()
        with pytest.raises(RuntimeError, match="backup in progress"):
            mem.add("should be blocked", domain="test")
        # Clean up
        mem.backup_release()

    def test_backup_release_removes_lock(self, mem: Memory):
        lock_path = mem.backup_prep()
        assert Path(lock_path).exists()
        mem.backup_release()
        assert not Path(lock_path).exists()

    def test_ingest_works_after_release(self, mem: Memory):
        mem.backup_prep()
        mem.backup_release()
        chunk_id = mem.add("text after release", domain="test", topic="backup")
        assert chunk_id is not None
        results = mem.search("text after release", domain="test")
        assert len(results) >= 1
