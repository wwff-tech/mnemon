"""Knowledge graph module — entity and triple management over SQLite."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from mnemon.db import Database

VALID_ENTITY_TYPES = {"person", "project", "system", "concept"}


class KnowledgeGraph:
    """High-level CRUD for entities and triples stored in a :class:`Database`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entity(
        self,
        name: str,
        type: str,
        properties: dict[str, Any] | None = None,
    ) -> int:
        """Insert a new entity and return its id.

        *type* must be one of: person, project, system, concept.
        """
        if type not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity type {type!r}. "
                f"Must be one of: {', '.join(sorted(VALID_ENTITY_TYPES))}"
            )
        props_json = json.dumps(properties) if properties else None
        cur = self.db.execute(
            "INSERT INTO entities (name, type, properties, created_at) VALUES (?, ?, ?, ?)",
            (name, type, props_json, datetime.now(UTC).isoformat()),
        )
        self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Look up an entity by *name*.  Returns a dict or ``None``."""
        row = self.db.fetchone("SELECT * FROM entities WHERE name = ?", (name,))
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "properties": json.loads(row["properties"]) if row["properties"] else None,
            "created_at": row["created_at"],
        }

    def get_or_create_entity(self, name: str, type: str) -> int:
        """Return the id of the entity called *name*, creating it if needed."""
        entity = self.get_entity(name)
        if entity is not None:
            entity_id: int = entity["id"]
            return entity_id
        return self.create_entity(name, type)

    # ------------------------------------------------------------------
    # Triple operations
    # ------------------------------------------------------------------

    def add(
        self,
        subject: str,
        predicate: str,
        object: str,
        valid_from: str,
        *,
        subject_type: str = "concept",
        object_type: str = "concept",
        confidence: float = 1.0,
        source: str | None = None,
    ) -> int:
        """Create a triple linking *subject* to *object* via *predicate*.

        Entities are auto-created when they do not yet exist.
        Returns the new triple id.
        """
        subject_id = self.get_or_create_entity(subject, subject_type)
        object_id = self.get_or_create_entity(object, object_type)
        cur = self.db.execute(
            """INSERT INTO triples
               (subject_id, predicate, object_id, object_val,
                valid_from, confidence, source, created_at)
               VALUES (?, ?, ?, NULL, ?, ?, ?, ?)""",
            (
                subject_id,
                predicate,
                object_id,
                valid_from,
                confidence,
                source,
                datetime.now(UTC).isoformat(),
            ),
        )
        self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def invalidate(
        self,
        subject: str,
        predicate: str,
        object: str,
        ended: str,
    ) -> bool:
        """Set *valid_to* on a currently-active triple.

        Returns ``True`` if a matching triple was found and updated.
        """
        cur = self.db.execute(
            """UPDATE triples
               SET valid_to = ?
               WHERE valid_to IS NULL
                 AND predicate = ?
                 AND subject_id = (SELECT id FROM entities WHERE name = ?)
                 AND object_id  = (SELECT id FROM entities WHERE name = ?)""",
            (ended, predicate, subject, object),
        )
        self.db.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query(self, entity_name: str, as_of: str | None = None) -> list[dict[str, Any]]:
        """Return triples where *entity_name* appears as subject or object.

        When *as_of* is given, only triples valid at that point in time are
        included (``valid_from <= as_of`` and either no ``valid_to`` or
        ``valid_to > as_of``).
        """
        if as_of is not None:
            sql = """
                SELECT s.name AS subject_name,
                       t.predicate,
                       o.name AS object_name,
                       t.valid_from,
                       t.valid_to,
                       t.confidence
                FROM triples t
                JOIN entities s ON t.subject_id = s.id
                JOIN entities o ON t.object_id  = o.id
                WHERE (s.name = ? OR o.name = ?)
                  AND t.valid_from <= ?
                  AND (t.valid_to IS NULL OR t.valid_to > ?)
            """
            rows = self.db.fetchall(sql, (entity_name, entity_name, as_of, as_of))
        else:
            sql = """
                SELECT s.name AS subject_name,
                       t.predicate,
                       o.name AS object_name,
                       t.valid_from,
                       t.valid_to,
                       t.confidence
                FROM triples t
                JOIN entities s ON t.subject_id = s.id
                JOIN entities o ON t.object_id  = o.id
                WHERE s.name = ? OR o.name = ?
            """
            rows = self.db.fetchall(sql, (entity_name, entity_name))

        return [
            {
                "subject_name": r["subject_name"],
                "predicate": r["predicate"],
                "object_name": r["object_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def timeline(self, entity_name: str) -> list[dict[str, Any]]:
        """Return *all* triples (including invalidated) for *entity_name*,
        ordered chronologically by ``valid_from``."""
        sql = """
            SELECT s.name AS subject_name,
                   t.predicate,
                   o.name AS object_name,
                   t.valid_from,
                   t.valid_to,
                   t.confidence
            FROM triples t
            JOIN entities s ON t.subject_id = s.id
            JOIN entities o ON t.object_id  = o.id
            WHERE s.name = ? OR o.name = ?
            ORDER BY t.valid_from ASC
        """
        rows = self.db.fetchall(sql, (entity_name, entity_name))
        return [
            {
                "subject_name": r["subject_name"],
                "predicate": r["predicate"],
                "object_name": r["object_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]
