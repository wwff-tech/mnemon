"""SQLite database module — schema management and query helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Sequence

_MIGRATIONS: dict[int, str] = {
    1: """
CREATE TABLE entities (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    properties  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE triples (
    id          INTEGER PRIMARY KEY,
    subject_id  INTEGER NOT NULL REFERENCES entities(id),
    predicate   TEXT NOT NULL,
    object_id   INTEGER REFERENCES entities(id),
    object_val  TEXT,
    valid_from  TEXT NOT NULL,
    valid_to    TEXT,
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX idx_triples_subject ON triples(subject_id);
CREATE INDEX idx_triples_valid   ON triples(valid_from, valid_to);

CREATE TABLE sources (
    id                  INTEGER PRIMARY KEY,
    path                TEXT NOT NULL UNIQUE,
    format              TEXT NOT NULL,
    content_hash        TEXT,
    first_ingested_at   TEXT NOT NULL,
    last_ingested_at    TEXT NOT NULL,
    last_event_at       TEXT,
    chunk_count         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE review_queue (
    id              INTEGER PRIMARY KEY,
    chunk_id        TEXT NOT NULL,
    guessed_domain  TEXT NOT NULL,
    guessed_topic   TEXT,
    confidence      REAL NOT NULL,
    raw_text        TEXT,
    queued_at       TEXT NOT NULL,
    resolved_at     TEXT,
    resolved_domain TEXT
);

CREATE INDEX idx_review_unresolved ON review_queue(resolved_at)
    WHERE resolved_at IS NULL;
""",
}


class Database:
    """Thin wrapper around a SQLite connection with automatic migrations."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            # Import default only when needed, keeping the module standalone-capable.
            from mnemon.config import load_config

            path = load_config().db_path

        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._ensure_migrations_table()
        self._run_migrations()

    # ------------------------------------------------------------------
    # Migration machinery
    # ------------------------------------------------------------------

    def _ensure_migrations_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _run_migrations(self) -> None:
        cur = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        current_version: int = cur.fetchone()[0]

        for version in sorted(_MIGRATIONS):
            if version <= current_version:
                continue
            self._conn.executescript(_MIGRATIONS[version])
            self._conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, datetime('now'))",
                (version,),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement and return the cursor."""
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_list: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        """Execute a SQL statement against each parameter set."""
        return self._conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """Execute a query and return the first row, or ``None``."""
        return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Execute a query and return all rows."""
        return self._conn.execute(sql, params).fetchall()

    def commit(self) -> None:
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
