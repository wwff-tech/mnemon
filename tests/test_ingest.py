"""Tests for the ingest pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from mnemon.config import MnemonConfig
from mnemon.db import Database
from mnemon.ingest import IngestPipeline, IngestResult
from mnemon.resolver import HeuristicResolver
from mnemon.vectorstore import ChromaStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl_fixture(path: Path) -> Path:
    """Write a minimal Claude JSONL session file and return its path."""
    filepath = path / "test_session.jsonl"
    messages = [
        {
            "type": "human",
            "content": "How do I deploy the auth service to production?",
            "timestamp": "2025-06-01T10:00:00Z",
        },
        {
            "type": "assistant",
            "content": "You can deploy the auth service using the CI/CD pipeline with kubectl.",
            "timestamp": "2025-06-01T10:00:05Z",
        },
        {
            "type": "human",
            "content": "What about the database migration?",
            "timestamp": "2025-06-01T10:01:00Z",
        },
        {
            "type": "assistant",
            "content": "Run the migration script before deploying: ./migrate.sh production",
            "timestamp": "2025-06-01T10:01:05Z",
        },
    ]
    with open(filepath, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return filepath


def _make_pipeline(tmp_path: Path) -> tuple[IngestPipeline, MnemonConfig]:
    """Build a full pipeline with real Database and ChromaStore."""
    base = tmp_path / "mnemon"
    base.mkdir(exist_ok=True)
    (base / "canonical").mkdir(exist_ok=True)
    (base / "chroma").mkdir(exist_ok=True)

    config = MnemonConfig(base_dir=base)
    db = Database(config.db_path)
    store = ChromaStore(
        persist_directory=str(config.chroma_dir),
        collection_name=config.chroma_collection,
    )
    resolver = HeuristicResolver()
    pipeline = IngestPipeline(store=store, db=db, resolver=resolver, config=config)
    return pipeline, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestConversation:
    """End-to-end conversation ingestion tests."""

    def test_ingest_conversation_e2e(self, tmp_path: Path) -> None:
        pipeline, config = _make_pipeline(tmp_path)
        fixture = _write_jsonl_fixture(tmp_path)

        result = pipeline.ingest_conversation(fixture, domain="engineering")

        assert isinstance(result, IngestResult)
        assert result.session_id == "test_session"
        assert result.chunks_added > 0
        assert result.domain is not None

        # Canonical file should exist
        canonical_path = config.canonical_dir / "test_session.json"
        assert canonical_path.exists()

        # Chunks should be retrievable from the store
        chunk_id = f"{result.session_id}_0"
        doc = pipeline.store.get(chunk_id)
        assert doc is not None
        assert doc.metadata["domain"] == "engineering"

    def test_incremental_skip(self, tmp_path: Path) -> None:
        pipeline, _config = _make_pipeline(tmp_path)
        fixture = _write_jsonl_fixture(tmp_path)

        first = pipeline.ingest_conversation(fixture, domain="engineering")
        assert first.chunks_added > 0

        second = pipeline.ingest_conversation(fixture, domain="engineering")
        # Same file, same hash — should skip entirely
        assert second.chunks_added == 0
        assert second.chunks_skipped == 0


    def test_reingest_changed_content_updates_vectors(self, tmp_path: Path) -> None:
        pipeline, _config = _make_pipeline(tmp_path)
        fixture = _write_jsonl_fixture(tmp_path)

        first = pipeline.ingest_conversation(fixture, domain="engineering")
        assert first.chunks_added > 0

        # Modify the file content
        with open(fixture, "a") as f:
            f.write(json.dumps({
                "type": "human",
                "content": "New question about billing?",
                "timestamp": "2025-06-01T10:02:00Z",
            }) + "\n")
            f.write(json.dumps({
                "type": "assistant",
                "content": "Billing uses Stripe integration.",
                "timestamp": "2025-06-01T10:02:05Z",
            }) + "\n")

        second = pipeline.ingest_conversation(fixture, domain="engineering")
        # Changed content should re-ingest, not skip
        assert second.chunks_added > first.chunks_added


class TestIngestFile:
    """File ingestion tests."""

    def test_ingest_file(self, tmp_path: Path) -> None:
        pipeline, _config = _make_pipeline(tmp_path)

        # Write a text file with multiple paragraphs
        text_file = tmp_path / "notes.txt"
        text_file.write_text(
            "Authentication uses JWT tokens for session management.\n\n"
            "The deploy pipeline runs on GitHub Actions with staging gates.\n\n"
            "Database migrations are managed by Alembic.\n"
        )

        result = pipeline.ingest_file(text_file, domain="engineering")

        assert isinstance(result, IngestResult)
        assert result.session_id is None
        assert result.chunks_added > 0
        assert result.domain == "engineering"

        # Verify first chunk exists in store
        chunk_id = "file_engineering_notes_0"
        doc = pipeline.store.get(chunk_id)
        assert doc is not None
        assert doc.metadata["domain"] == "engineering"


class TestIngestText:
    """Direct text ingestion tests."""

    def test_ingest_text_returns_id(self, tmp_path: Path) -> None:
        pipeline, _config = _make_pipeline(tmp_path)

        chunk_id = pipeline.ingest_text(
            "We decided to use PostgreSQL for the billing service.",
            domain="engineering",
            topic="billing",
        )

        assert chunk_id.startswith("text_")
        doc = pipeline.store.get(chunk_id)
        assert doc is not None
        assert "PostgreSQL" in doc.text

    def test_dedup_same_text(self, tmp_path: Path) -> None:
        pipeline, _config = _make_pipeline(tmp_path)

        text = "The auth service should use Redis for token caching."

        id1 = pipeline.ingest_text(text, domain="engineering")
        id2 = pipeline.ingest_text(text, domain="engineering")

        assert id1 == id2
        # Only one document in the store for this content
        assert pipeline.store.count() == 1


class TestSourcesTable:
    """Verify the sources table is updated correctly."""

    def test_sources_row_created(self, tmp_path: Path) -> None:
        pipeline, _config = _make_pipeline(tmp_path)
        fixture = _write_jsonl_fixture(tmp_path)

        result = pipeline.ingest_conversation(fixture, domain="engineering")

        row = pipeline.db.fetchone(
            "SELECT * FROM sources WHERE path = ?", (str(fixture),)
        )
        assert row is not None
        assert row["format"] == "claude_jsonl"
        assert row["content_hash"] is not None
        assert row["chunk_count"] == result.chunks_added


class TestRegenerateL1:
    """Verify L1 regeneration is callable."""

    def test_regenerate_l1_no_error(self, tmp_path: Path) -> None:
        pipeline, config = _make_pipeline(tmp_path)

        # Ingest something so the store is not empty
        pipeline.ingest_text(
            "Important: always run migrations before deploy.",
            domain="operations",
            topic="deploy",
        )

        # Should not raise
        pipeline.regenerate_l1()

        # L1 cache file should exist
        assert config.l1_cache_path.exists()
