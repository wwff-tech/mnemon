"""Public API for Mnemon — the ``Memory`` class."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReviewQueue:
    """Helper exposing the review-queue table through the Memory API."""

    def __init__(self, db: Any, store: Any) -> None:
        self._db = db
        self._store = store

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return unresolved review-queue entries."""
        rows = self._db.fetchall(
            "SELECT * FROM review_queue WHERE resolved_at IS NULL ORDER BY queued_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "id": r["id"],
                "chunk_id": r["chunk_id"],
                "guessed_domain": r["guessed_domain"],
                "guessed_topic": r["guessed_topic"],
                "confidence": r["confidence"],
                "raw_text": r["raw_text"],
                "queued_at": r["queued_at"],
            }
            for r in rows
        ]

    def resolve(
        self,
        chunk_id: str,
        domain: str,
        topic: str | None = None,
    ) -> None:
        """Mark a review-queue entry as resolved and update chunk metadata."""
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "UPDATE review_queue "
            "SET resolved_at = ?, resolved_domain = ?, resolved_topic = ? "
            "WHERE chunk_id = ? AND resolved_at IS NULL",
            (now, domain, topic, chunk_id),
        )
        self._db.commit()

        # Update chunk metadata in the vector store.
        existing = self._store.get(chunk_id)
        if existing is not None:
            meta = dict(existing.metadata)
            meta["domain"] = domain
            if topic is not None:
                meta["topic"] = topic
            meta["low_confidence_domain"] = False
            # ChromaDB supports upsert: delete + re-add.
            self._store.delete(chunk_id)
            self._store.add(chunk_id, existing.text, meta)


class Memory:
    """Single public entry-point for the Mnemon memory system.

    Usage::

        mem = Memory()                       # config from ~/.mnemon/config.json
        mem.add("some text", domain="hermes")
        results = mem.search("auth decisions", domain="hermes")
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        from mnemon.config import load_config
        from mnemon.db import Database
        from mnemon.ingest import IngestPipeline
        from mnemon.kg import KnowledgeGraph
        from mnemon.memory import MemoryStack
        from mnemon.resolver import get_resolver
        from mnemon.vectorstore import ChromaStore

        # Build config.
        overrides: dict[str, Any] | None = None
        if base_dir is not None:
            overrides = {"base_dir": str(base_dir)}
        cfg_path = Path(config_path) if config_path is not None else None
        self._config = load_config(config_path=cfg_path, overrides=overrides)
        self._config.ensure_dirs()

        # Core components.
        self._db = Database(self._config.db_path)
        self._store = ChromaStore(
            persist_directory=self._config.chroma_dir,
            collection_name=self._config.chroma_collection,
        )
        self._resolver = get_resolver(
            self._config.default_resolver,
            **( {"domain_map": self._config.domain_map, "db": self._db}
                if self._config.default_resolver == "heuristic"
                else {}),
        )
        self._memory_stack = MemoryStack(self._store, self._config)
        self.kg = KnowledgeGraph(self._db)
        self._ingest = IngestPipeline(
            store=self._store,
            db=self._db,
            resolver=self._resolver,
            config=self._config,
        )
        self._review = ReviewQueue(self._db, self._store)

    # ── properties ──────────────────────────────────────────────

    @property
    def review(self) -> ReviewQueue:
        """Access the review queue."""
        return self._review

    # ── core methods ────────────────────────────────────────────

    def add(
        self,
        text: str,
        domain: str | None = None,
        topic: str | None = None,
        importance: float = 0.5,
        source: str | None = None,
        resolver: str | None = None,
    ) -> str:
        """Ingest a text chunk. Returns the chunk ID."""
        resolver_mode = "heuristic" if resolver == "heuristic" else None
        return self._ingest.ingest_text(
            text=text,
            domain=domain,
            topic=topic,
            importance=importance,
            source=source,
            resolver_mode=resolver_mode,
        )

    def search(
        self,
        query: str,
        domain: str | None = None,
        topic: str | None = None,
        n: int = 5,
    ) -> list[Any]:
        """Search memory. Falls back to L3 (unfiltered) only when L2 returns nothing."""
        results = self._memory_stack.search_l2(query, domain=domain, topic=topic, n=n)
        if not results:
            results = self._memory_stack.search_l3(query, n=n)
        return results

    def wake_up(self, domain: str | None = None) -> str:
        """Load identity + top-of-mind context."""
        return self._memory_stack.wake_up(domain=domain)

    def status(self) -> str:
        """Return identity + top-of-mind + protocol reminder."""
        return self._memory_stack.status()

    def ingest(
        self,
        path: str | Path,
        domain: str | None = None,
        topic: str | None = None,
        importance: float = 0.5,
    ) -> Any:
        """Ingest a file or conversation. Auto-detects format by extension."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".json"):
            return self._ingest.ingest_conversation(
                path=path, domain=domain, topic=topic, importance=importance,
            )
        return self._ingest.ingest_file(
            path=path, domain=domain or "general", topic=topic, importance=importance,
        )

    def list_domains(self) -> list[dict[str, Any]]:
        """Return unique domains with document counts."""
        metas = self._store.list_metadata()
        counts: dict[str, int] = {}
        for meta in metas:
            domain = meta.get("domain", "unknown")
            counts[domain] = counts.get(domain, 0) + 1
        return [{"domain": d, "count": c} for d, c in sorted(counts.items())]

    def list_topics(self, domain: str) -> list[dict[str, Any]]:
        """Return unique topics for a domain with document counts."""
        metas = self._store.list_metadata(where={"domain": domain})
        counts: dict[str, int] = {}
        for meta in metas:
            topic = meta.get("topic", "")
            counts[topic] = counts.get(topic, 0) + 1
        return [{"topic": t, "count": c} for t, c in sorted(counts.items())]

    def check_duplicate(self, text: str) -> bool:
        """Return True if text has already been ingested (by content hash)."""
        from mnemon.chunking import compute_chunk_hash

        content_hash = compute_chunk_hash(text)
        chunk_id = f"text_{content_hash[:16]}"
        return self._store.get(chunk_id) is not None

    # ── lifecycle ───────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
