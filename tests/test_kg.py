"""Tests for the KnowledgeGraph module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemon.db import Database
from mnemon.kg import KnowledgeGraph


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    db = Database(path=tmp_path / "test.db")
    return KnowledgeGraph(db)


class TestEntityCRUD:
    def test_create_and_get_entity(self, kg: KnowledgeGraph) -> None:
        eid = kg.create_entity("Alice", "person", {"role": "engineer"})
        assert isinstance(eid, int)

        entity = kg.get_entity("Alice")
        assert entity is not None
        assert entity["name"] == "Alice"
        assert entity["type"] == "person"
        assert entity["properties"] == {"role": "engineer"}

    def test_get_entity_returns_none_for_missing(self, kg: KnowledgeGraph) -> None:
        assert kg.get_entity("Nobody") is None

    def test_get_or_create_is_idempotent(self, kg: KnowledgeGraph) -> None:
        id1 = kg.get_or_create_entity("Bob", "person")
        id2 = kg.get_or_create_entity("Bob", "person")
        assert id1 == id2

    def test_invalid_entity_type_rejected(self, kg: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="Invalid entity type"):
            kg.create_entity("Bad", "banana")

    def test_invalid_type_via_get_or_create(self, kg: KnowledgeGraph) -> None:
        with pytest.raises(ValueError, match="Invalid entity type"):
            kg.get_or_create_entity("Bad", "banana")


class TestTriples:
    def test_add_triple(self, kg: KnowledgeGraph) -> None:
        tid = kg.add(
            "Alice", "works_on", "Mnemon",
            valid_from="2026-01-01",
            subject_type="person",
            object_type="project",
        )
        assert isinstance(tid, int)

        # Both entities should exist now.
        assert kg.get_entity("Alice") is not None
        assert kg.get_entity("Mnemon") is not None

    def test_query_returns_triples(self, kg: KnowledgeGraph) -> None:
        kg.add("Alice", "works_on", "Mnemon", valid_from="2026-01-01",
               subject_type="person", object_type="project")
        kg.add("Bob", "works_on", "Mnemon", valid_from="2026-02-01",
               subject_type="person", object_type="project")

        results = kg.query("Mnemon")
        assert len(results) == 2
        predicates = {r["predicate"] for r in results}
        assert predicates == {"works_on"}

    def test_invalidate_triple(self, kg: KnowledgeGraph) -> None:
        kg.add("Alice", "works_on", "Mnemon", valid_from="2026-01-01",
               subject_type="person", object_type="project")

        ok = kg.invalidate("Alice", "works_on", "Mnemon", ended="2026-03-01")
        assert ok is True

        # Invalidating again should return False (already has valid_to).
        ok2 = kg.invalidate("Alice", "works_on", "Mnemon", ended="2026-04-01")
        assert ok2 is False

    def test_invalidate_nonexistent_returns_false(self, kg: KnowledgeGraph) -> None:
        assert kg.invalidate("X", "rel", "Y", ended="2026-01-01") is False


class TestPointInTimeQuery:
    def test_as_of_filters_correctly(self, kg: KnowledgeGraph) -> None:
        kg.add("Alice", "works_on", "Mnemon", valid_from="2026-01",
               subject_type="person", object_type="project")
        kg.invalidate("Alice", "works_on", "Mnemon", ended="2026-03")

        # During the valid window the triple is visible.
        visible = kg.query("Alice", as_of="2026-02")
        assert len(visible) == 1
        assert visible[0]["subject_name"] == "Alice"
        assert visible[0]["object_name"] == "Mnemon"

        # After invalidation it is gone.
        gone = kg.query("Alice", as_of="2026-04")
        assert len(gone) == 0


class TestTimeline:
    def test_timeline_chronological_with_invalidated(self, kg: KnowledgeGraph) -> None:
        kg.add("Alice", "works_on", "ProjectA", valid_from="2026-01",
               subject_type="person", object_type="project")
        kg.add("Alice", "works_on", "ProjectB", valid_from="2026-03",
               subject_type="person", object_type="project")
        kg.invalidate("Alice", "works_on", "ProjectA", ended="2026-03")

        tl = kg.timeline("Alice")
        assert len(tl) == 2
        # Chronological order.
        assert tl[0]["valid_from"] <= tl[1]["valid_from"]
        # First entry is invalidated.
        assert tl[0]["valid_to"] == "2026-03"
        # Second entry is still active.
        assert tl[1]["valid_to"] is None
