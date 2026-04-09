"""Integration tests for the Memory public API."""

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


class TestMemoryLifecycle:
    def test_creates_without_error(self, tmp_path: Path):
        m = Memory(base_dir=tmp_path / "mnemon")
        m.close()

    def test_context_manager(self, tmp_path: Path):
        with Memory(base_dir=tmp_path / "mnemon") as m:
            assert m is not None


class TestAddAndSearch:
    def test_round_trip(self, mem: Memory):
        mem.add("auth token rotation policy for hermes", domain="hermes", topic="auth")
        results = mem.search("auth token", domain="hermes")
        assert len(results) >= 1
        assert "auth" in results[0].text.lower()

    def test_search_falls_back_to_l3(self, mem: Memory):
        mem.add("unique canary phrase xyz123", domain="alpha", topic="misc")
        # Search with a different domain so L2 returns nothing, triggering L3 fallback.
        results = mem.search("canary phrase xyz123", domain="nonexistent")
        assert len(results) >= 1


class TestListDomains:
    def test_returns_added_domains(self, mem: Memory):
        mem.add("first item", domain="proj_a", topic="t1")
        mem.add("second item", domain="proj_b", topic="t2")
        domains = mem.list_domains()
        domain_names = [d["domain"] for d in domains]
        assert "proj_a" in domain_names
        assert "proj_b" in domain_names

    def test_counts_are_correct(self, mem: Memory):
        mem.add("item one", domain="proj_c", topic="t1")
        mem.add("item two different text", domain="proj_c", topic="t1")
        domains = mem.list_domains()
        proj_c = [d for d in domains if d["domain"] == "proj_c"]
        assert proj_c[0]["count"] == 2


class TestListTopics:
    def test_returns_topics_for_domain(self, mem: Memory):
        mem.add("design doc", domain="hermes", topic="design")
        mem.add("deploy plan", domain="hermes", topic="deploy")
        topics = mem.list_topics("hermes")
        topic_names = [t["topic"] for t in topics]
        assert "design" in topic_names
        assert "deploy" in topic_names


class TestCheckDuplicate:
    def test_detects_duplicate(self, mem: Memory):
        text = "a unique piece of text for dedup testing"
        assert mem.check_duplicate(text) is False
        mem.add(text, domain="test", topic="dedup")
        assert mem.check_duplicate(text) is True


class TestKnowledgeGraph:
    def test_add_and_query(self, mem: Memory):
        mem.kg.add("hermes", "uses", "rabbitmq", valid_from="2026-01-01")
        results = mem.kg.query("hermes")
        assert len(results) == 1
        assert results[0]["predicate"] == "uses"
        assert results[0]["object_name"] == "rabbitmq"

    def test_query_as_of(self, mem: Memory):
        mem.kg.add("hermes", "uses", "rabbitmq", valid_from="2026-01-01")
        mem.kg.invalidate("hermes", "uses", "rabbitmq", ended="2026-04-01")
        # Before invalidation date — should find it.
        results = mem.kg.query("hermes", as_of="2026-02-01")
        assert len(results) == 1
        # After invalidation date — should not find it.
        results = mem.kg.query("hermes", as_of="2026-05-01")
        assert len(results) == 0

    def test_timeline(self, mem: Memory):
        mem.kg.add("hermes", "uses", "rabbitmq", valid_from="2026-01-01")
        mem.kg.add("hermes", "uses", "kafka", valid_from="2026-04-01")
        tl = mem.kg.timeline("hermes")
        assert len(tl) == 2
        assert tl[0]["valid_from"] <= tl[1]["valid_from"]


class TestWakeUpAndStatus:
    def test_wake_up_returns_string(self, mem: Memory):
        result = mem.wake_up()
        assert isinstance(result, str)

    def test_wake_up_with_domain(self, mem: Memory):
        result = mem.wake_up(domain="hermes")
        assert isinstance(result, str)

    def test_status_includes_protocol(self, mem: Memory):
        result = mem.status()
        assert isinstance(result, str)
        assert "mnemon_search" in result
