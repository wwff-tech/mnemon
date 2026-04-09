"""Ingest pipeline — parse, chunk, deduplicate, and embed content."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mnemon import canonical, chunking, parsers
from mnemon.chunking import compute_chunk_hash
from mnemon.config import MnemonConfig
from mnemon.db import Database
from mnemon.memory import MemoryStack
from mnemon.resolver import DomainResolver, HeuristicResolver
from mnemon.vectorstore import VectorStore


@dataclass
class IngestResult:
    """Summary returned after ingesting a single source."""

    session_id: str | None
    chunks_added: int
    chunks_skipped: int
    domain: str | None
    topic: str | None


def _file_content_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestPipeline:
    """Orchestrates parsing, chunking, dedup, and storage of content."""

    def __init__(
        self,
        store: VectorStore,
        db: Database,
        resolver: DomainResolver,
        config: MnemonConfig,
    ) -> None:
        self.store = store
        self.db = db
        self.resolver = resolver
        self.config = config

    # ------------------------------------------------------------------
    # Conversation ingestion (JSONL / JSON session files)
    # ------------------------------------------------------------------

    def ingest_conversation(
        self,
        path: str | Path,
        domain: str | None = None,
        topic: str | None = None,
        importance: float = 0.5,
    ) -> IngestResult:
        """Parse a conversation file, chunk it, and add to the vector store."""
        path = Path(path)

        # 1. Parse
        session = parsers.parse(path)

        # 2. Write canonical
        canonical.write_canonical(session, self.config.canonical_dir)

        # 3. Content-hash the source file for incremental skip
        content_hash = _file_content_hash(path)

        existing = self.db.fetchone(
            "SELECT content_hash FROM sources WHERE path = ?", (str(path),)
        )
        if existing and existing["content_hash"] == content_hash:
            return IngestResult(
                session_id=session.session_id,
                chunks_added=0,
                chunks_skipped=0,
                domain=domain,
                topic=topic,
            )

        # 4. Upsert sources row
        now = _now_iso()
        self.db.execute(
            """
            INSERT OR REPLACE INTO sources
                (path, format, content_hash, first_ingested_at, last_ingested_at,
                 last_event_at, chunk_count)
            VALUES (
                ?, ?, ?, COALESCE(
                    (SELECT first_ingested_at FROM sources WHERE path = ?), ?
                ), ?, ?, 0
            )
            """,
            (
                str(path),
                session.source_format,
                content_hash,
                str(path),
                now,
                now,
                session.last_event_at,
            ),
        )
        self.db.commit()

        # 5. Resolve domain / topic
        metadata: dict[str, Any] = {"source": str(path)}
        if domain:
            metadata["domain"] = domain
        if topic:
            metadata["topic"] = topic

        first_text = ""
        if session.exchanges:
            ex = session.exchanges[0]
            first_text = ex.get("user", "") + "\n" + ex.get("assistant", "")

        resolved = self.resolver.resolve(first_text, metadata)
        domain = resolved.domain
        topic = resolved.topic

        # 6. Chunk
        chunks = chunking.chunk_exchanges(session.exchanges)

        # 7. Add chunks, deduplicating by ID
        added = 0
        skipped = 0
        for chunk in chunks:
            chunk_id = f"{session.session_id}_{chunk.chunk_index}"
            if self.store.get(chunk_id) is not None:
                skipped += 1
                continue

            meta: dict[str, Any] = {
                "domain": domain,
                "topic": topic or "",
                "source": str(path),
                "timestamp": chunk.metadata.get("timestamp", ""),
                "importance": importance,
                "chunk_index": chunk.chunk_index,
                "exchange_index": chunk.exchange_index if chunk.exchange_index is not None else -1,
                "session_id": session.session_id,
                "low_confidence_domain": resolved.low_confidence,
            }
            self.store.add(chunk_id, chunk.text, meta)
            added += 1

        # 8. Update chunk_count
        self.db.execute(
            "UPDATE sources SET chunk_count = ? WHERE path = ?",
            (added + skipped, str(path)),
        )
        self.db.commit()

        return IngestResult(
            session_id=session.session_id,
            chunks_added=added,
            chunks_skipped=skipped,
            domain=domain,
            topic=topic,
        )

    # ------------------------------------------------------------------
    # File ingestion (plain text / markdown)
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        path: str | Path,
        domain: str,
        topic: str | None = None,
        importance: float = 0.5,
    ) -> IngestResult:
        """Read a text file, chunk by paragraphs, and add to the vector store."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        # Resolve topic from path if not provided
        if topic is None:
            metadata: dict[str, Any] = {"domain": domain, "source": str(path)}
            resolved = self.resolver.resolve(text, metadata)
            topic = resolved.topic

        # Chunk
        chunks = chunking.chunk_paragraphs(text)
        stem = path.stem

        added = 0
        skipped = 0
        for chunk in chunks:
            chunk_id = f"file_{domain}_{stem}_{chunk.chunk_index}"
            if self.store.get(chunk_id) is not None:
                skipped += 1
                continue

            meta: dict[str, Any] = {
                "domain": domain,
                "topic": topic or "",
                "source": str(path),
                "importance": importance,
                "chunk_index": chunk.chunk_index,
            }
            self.store.add(chunk_id, chunk.text, meta)
            added += 1

        return IngestResult(
            session_id=None,
            chunks_added=added,
            chunks_skipped=skipped,
            domain=domain,
            topic=topic,
        )

    # ------------------------------------------------------------------
    # Direct text ingestion
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        domain: str | None = None,
        topic: str | None = None,
        importance: float = 0.5,
        source: str | None = None,
        resolver_mode: str | None = None,
    ) -> str:
        """Ingest a single text chunk directly. Returns the chunk ID."""
        content_hash = compute_chunk_hash(text)
        chunk_id = f"text_{content_hash[:16]}"

        # Dedup
        if self.store.get(chunk_id) is not None:
            return chunk_id

        # Resolve domain / topic
        metadata: dict[str, Any] = {}
        if domain:
            metadata["domain"] = domain
        if topic:
            metadata["topic"] = topic
        if source:
            metadata["source"] = source

        resolver = self.resolver
        if resolver_mode == "heuristic":
            resolver = HeuristicResolver()

        resolved = resolver.resolve(text, metadata)

        meta: dict[str, Any] = {
            "domain": resolved.domain,
            "topic": resolved.topic or "",
            "source": source or "",
            "importance": importance,
            "content_hash": content_hash,
        }
        self.store.add(chunk_id, text, meta)
        return chunk_id

    # ------------------------------------------------------------------
    # L1 regeneration
    # ------------------------------------------------------------------

    def regenerate_l1(self) -> None:
        """Rebuild the L1 top-of-mind summary from current store contents."""
        stack = MemoryStack(self.store, self.config)
        stack.generate_l1()
