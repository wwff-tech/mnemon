"""Tests for the MCP server module."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import mnemon.server as server_module
from mnemon.api import Memory
from mnemon.server import mcp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem(tmp_path: Path):
    """Create a Memory backed by a temp directory and patch get_memory()."""
    m = Memory(base_dir=tmp_path / "mnemon")
    original = server_module._memory
    server_module._memory = m
    yield m
    server_module._memory = original
    m.close()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = [
    "mnemon_status",
    "mnemon_wake_up",
    "mnemon_search",
    "mnemon_list_domains",
    "mnemon_list_topics",
    "mnemon_kg_query",
    "mnemon_kg_timeline",
    "mnemon_add",
    "mnemon_kg_add",
    "mnemon_kg_invalidate",
    "mnemon_check_duplicate",
]


class TestToolRegistration:
    def test_all_tools_registered(self):
        tools = asyncio.run(mcp.list_tools())
        registered_names = {t.name for t in tools}
        for name in EXPECTED_TOOLS:
            assert name in registered_names, f"Tool {name!r} not registered"

    def test_exactly_11_tools(self):
        tools = asyncio.run(mcp.list_tools())
        assert len(tools) == 11


# ---------------------------------------------------------------------------
# Direct function tests
# ---------------------------------------------------------------------------


class TestStatus:
    def test_returns_string(self, mem: Memory):
        result = server_module.mnemon_status()
        assert isinstance(result, str)


class TestWakeUp:
    def test_returns_string(self, mem: Memory):
        result = server_module.mnemon_wake_up()
        assert isinstance(result, str)

    def test_with_domain(self, mem: Memory):
        result = server_module.mnemon_wake_up(domain="test")
        assert isinstance(result, str)


class TestAddAndSearch:
    def test_round_trip(self, mem: Memory):
        add_result = server_module.mnemon_add(
            text="auth token rotation policy for hermes",
            domain="hermes",
            topic="auth",
        )
        parsed = json.loads(add_result)
        assert parsed["ok"] is True
        assert "chunk_id" in parsed

        search_result = server_module.mnemon_search(query="auth token", domain="hermes")
        results = json.loads(search_result)
        assert len(results) >= 1
        assert "auth" in results[0]["text"].lower()


class TestListDomains:
    def test_returns_json_list(self, mem: Memory):
        mem.add("item", domain="proj_a", topic="t1")
        result = server_module.mnemon_list_domains()
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        domain_names = [d["domain"] for d in parsed]
        assert "proj_a" in domain_names


class TestListTopics:
    def test_returns_json_list(self, mem: Memory):
        mem.add("item", domain="proj_b", topic="design")
        result = server_module.mnemon_list_topics(domain="proj_b")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        topic_names = [t["topic"] for t in parsed]
        assert "design" in topic_names


class TestKnowledgeGraph:
    def test_kg_add_and_query(self, mem: Memory):
        add_result = server_module.mnemon_kg_add(
            subject="hermes",
            predicate="uses",
            object="rabbitmq",
            valid_from="2026-01-01",
        )
        parsed = json.loads(add_result)
        assert parsed["ok"] is True
        assert "edge_id" in parsed

        query_result = server_module.mnemon_kg_query(entity="hermes")
        facts = json.loads(query_result)
        assert len(facts) == 1
        assert facts[0]["predicate"] == "uses"

    def test_kg_timeline(self, mem: Memory):
        mem.kg.add("hermes", "uses", "rabbitmq", valid_from="2026-01-01")
        mem.kg.add("hermes", "uses", "kafka", valid_from="2026-04-01")
        result = server_module.mnemon_kg_timeline(entity="hermes")
        tl = json.loads(result)
        assert len(tl) == 2

    def test_kg_invalidate(self, mem: Memory):
        mem.kg.add("hermes", "uses", "rabbitmq", valid_from="2026-01-01")
        result = server_module.mnemon_kg_invalidate(
            subject="hermes",
            predicate="uses",
            object="rabbitmq",
            ended="2026-04-01",
        )
        parsed = json.loads(result)
        assert parsed["ok"] is True

    def test_kg_invalidate_missing(self, mem: Memory):
        result = server_module.mnemon_kg_invalidate(
            subject="nonexistent",
            predicate="uses",
            object="nothing",
            ended="2026-04-01",
        )
        parsed = json.loads(result)
        assert parsed["ok"] is False


class TestCheckDuplicate:
    def test_not_duplicate(self, mem: Memory):
        result = server_module.mnemon_check_duplicate(text="unique text for dedup test")
        parsed = json.loads(result)
        assert parsed["is_duplicate"] is False

    def test_is_duplicate_after_add(self, mem: Memory):
        text = "a unique piece of text for dedup testing"
        mem.add(text, domain="test", topic="dedup")
        result = server_module.mnemon_check_duplicate(text=text)
        parsed = json.loads(result)
        assert parsed["is_duplicate"] is True
