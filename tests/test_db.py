"""Tests for the Database module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemon.db import Database

EXPECTED_TABLES = {"entities", "triples", "sources", "review_queue", "schema_migrations"}


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(path=tmp_path / "test.db")


class TestMigrations:
    def test_all_tables_exist(self, db: Database) -> None:
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names = {row["name"] for row in rows}
        assert table_names == EXPECTED_TABLES

    def test_idempotent(self, tmp_path: Path) -> None:
        """Opening the database twice must not raise."""
        path = tmp_path / "idem.db"
        db1 = Database(path=path)
        db1.close()
        db2 = Database(path=path)
        db2.close()

    def test_version_tracking(self, db: Database) -> None:
        rows = db.fetchall("SELECT version FROM schema_migrations ORDER BY version")
        versions = [row["version"] for row in rows]
        assert versions == [1]

    def test_version_not_duplicated_on_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.db"
        Database(path=path).close()
        db = Database(path=path)
        rows = db.fetchall("SELECT version FROM schema_migrations")
        assert len(rows) == 1
        db.close()


class TestHelpers:
    def test_execute_and_fetchone(self, db: Database) -> None:
        db.execute(
            "INSERT INTO entities (name, type, created_at) VALUES (?, ?, ?)",
            ("Alice", "person", "2025-01-01T00:00:00Z"),
        )
        db.commit()
        row = db.fetchone("SELECT * FROM entities WHERE name = ?", ("Alice",))
        assert row is not None
        assert row["type"] == "person"

    def test_fetchall(self, db: Database) -> None:
        db.execute(
            "INSERT INTO entities (name, type, created_at) VALUES (?, ?, ?)",
            ("A", "person", "2025-01-01T00:00:00Z"),
        )
        db.execute(
            "INSERT INTO entities (name, type, created_at) VALUES (?, ?, ?)",
            ("B", "concept", "2025-01-02T00:00:00Z"),
        )
        db.commit()
        rows = db.fetchall("SELECT * FROM entities ORDER BY name")
        assert len(rows) == 2

    def test_executemany(self, db: Database) -> None:
        params = [
            ("X", "project", "2025-03-01T00:00:00Z"),
            ("Y", "system", "2025-03-02T00:00:00Z"),
        ]
        db.executemany(
            "INSERT INTO entities (name, type, created_at) VALUES (?, ?, ?)", params
        )
        db.commit()
        rows = db.fetchall("SELECT * FROM entities")
        assert len(rows) == 2

    def test_fetchone_returns_none_for_missing(self, db: Database) -> None:
        row = db.fetchone("SELECT * FROM entities WHERE id = ?", (9999,))
        assert row is None


class TestContextManager:
    def test_context_manager(self, tmp_path: Path) -> None:
        path = tmp_path / "ctx.db"
        with Database(path=path) as db:
            db.execute(
                "INSERT INTO entities (name, type, created_at) VALUES (?, ?, ?)",
                ("Z", "concept", "2025-06-01T00:00:00Z"),
            )
            db.commit()
        # Connection is closed — further ops should fail.
        with pytest.raises(Exception):
            db.execute("SELECT 1")
