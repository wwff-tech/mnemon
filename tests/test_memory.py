"""Tests for mnemon.memory."""

from __future__ import annotations

from typing import Any

import pytest

from mnemon.config import MnemonConfig
from mnemon.memory import PROTOCOL, MemoryStack
from mnemon.vectorstore import SearchResult, VectorStore


# ── Fake store ──────────────────────────────────────────────────


class FakeStore(VectorStore):
    """In-memory VectorStore for testing."""

    def __init__(self) -> None:
        self._docs: dict[str, tuple[str, dict[str, Any]]] = {}

    def add(self, id: str, text: str, metadata: dict[str, Any]) -> None:
        meta = {**metadata, "id": id}
        self._docs[id] = (text, meta)

    def search(
        self,
        query: str,
        n: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for doc_id, (text, meta) in self._docs.items():
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            results.append(
                SearchResult(id=doc_id, text=text, metadata=meta, score=0.9)
            )
        return results[:n]

    def delete(self, id: str) -> None:
        self._docs.pop(id, None)

    def get(self, id: str) -> SearchResult | None:
        if id not in self._docs:
            return None
        text, meta = self._docs[id]
        return SearchResult(id=id, text=text, metadata=meta, score=0.0)

    def list_metadata(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        metas: list[dict[str, Any]] = []
        for text, meta in self._docs.values():
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            metas.append(meta)
        return metas[:limit]


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def stack(store: FakeStore, config: MnemonConfig) -> MemoryStack:
    return MemoryStack(store=store, config=config)


# ── L0 ─────────────────────────────────────────────────────────


class TestLoadL0:
    def test_missing_identity_returns_empty(self, stack: MemoryStack) -> None:
        assert stack.load_l0() == ""

    def test_reads_identity_file(self, stack: MemoryStack, config: MnemonConfig) -> None:
        config.identity_path.write_text("I am Mnemon.", encoding="utf-8")
        assert stack.load_l0() == "I am Mnemon."


# ── L1 generation ──────────────────────────────────────────────


class TestGenerateL1:
    def test_top_items_selected_by_importance(
        self, stack: MemoryStack, store: FakeStore
    ) -> None:
        store.add("low", "low importance", {"importance": 0.1, "domain": "d"})
        store.add("high", "high importance", {"importance": 0.9, "domain": "d"})
        store.add("mid", "mid importance", {"importance": 0.5, "domain": "d"})

        text = stack.generate_l1()
        # high should come before mid, low may be cut off by ordering
        assert "high importance" in text
        assert "mid importance" in text

    def test_grouped_by_domain(
        self, stack: MemoryStack, store: FakeStore
    ) -> None:
        store.add("a", "alpha text", {"importance": 0.9, "domain": "work"})
        store.add("b", "beta text", {"importance": 0.8, "domain": "personal"})

        text = stack.generate_l1()
        assert "## work" in text
        assert "## personal" in text

    def test_cache_written_to_file(
        self, stack: MemoryStack, store: FakeStore, config: MnemonConfig
    ) -> None:
        store.add("x", "some text", {"importance": 0.7, "domain": "d"})
        stack.generate_l1()
        assert config.l1_cache_path.exists()
        assert "some text" in config.l1_cache_path.read_text(encoding="utf-8")

    def test_respects_max_chars_limit(
        self, stack: MemoryStack, store: FakeStore, config: MnemonConfig
    ) -> None:
        config.l1_max_chars = 100
        for i in range(20):
            store.add(
                f"item-{i}",
                f"text number {i} " * 10,
                {"importance": 0.9, "domain": "d"},
            )
        text = stack.generate_l1()
        assert len(text) <= config.l1_max_chars

    def test_items_truncated_to_item_max_chars(
        self, stack: MemoryStack, store: FakeStore, config: MnemonConfig
    ) -> None:
        config.l1_item_max_chars = 20
        store.add("long", "A" * 500, {"importance": 0.9, "domain": "d"})
        text = stack.generate_l1()
        # The item line is "- " + truncated text, so the A's should be ≤20
        for line in text.splitlines():
            if line.startswith("- "):
                content = line[2:]
                assert len(content) <= 20


# ── L1 loading ──────────────────────────────────────────────────


class TestLoadL1:
    def test_missing_cache_returns_empty(self, stack: MemoryStack) -> None:
        assert stack.load_l1() == ""

    def test_reads_cache(self, stack: MemoryStack, config: MnemonConfig) -> None:
        config.l1_cache_path.write_text("cached stuff", encoding="utf-8")
        assert stack.load_l1() == "cached stuff"


# ── L2 search ──────────────────────────────────────────────────


class TestSearchL2:
    def test_with_domain_filter(
        self, stack: MemoryStack, store: FakeStore
    ) -> None:
        store.add("a", "work item", {"domain": "work"})
        store.add("b", "play item", {"domain": "play"})

        results = stack.search_l2("item", domain="work")
        assert all(r.metadata["domain"] == "work" for r in results)

    def test_truncates_text(
        self, stack: MemoryStack, store: FakeStore
    ) -> None:
        store.add("a", "X" * 500, {"domain": "d"})
        results = stack.search_l2("X")
        assert len(results[0].text) == 300


# ── L3 search ──────────────────────────────────────────────────


class TestSearchL3:
    def test_no_filters(
        self, stack: MemoryStack, store: FakeStore
    ) -> None:
        store.add("a", "alpha", {"domain": "d"})
        store.add("b", "beta", {"domain": "e"})

        results = stack.search_l3("anything")
        assert len(results) == 2

    def test_no_truncation(
        self, stack: MemoryStack, store: FakeStore
    ) -> None:
        long_text = "Y" * 500
        store.add("a", long_text, {"domain": "d"})
        results = stack.search_l3("Y")
        assert results[0].text == long_text


# ── wake_up ─────────────────────────────────────────────────────


class TestWakeUp:
    def test_combines_l0_and_l1(
        self, stack: MemoryStack, config: MnemonConfig
    ) -> None:
        config.identity_path.write_text("I am Mnemon.", encoding="utf-8")
        config.l1_cache_path.write_text("## work\n- stuff", encoding="utf-8")

        result = stack.wake_up()
        assert "I am Mnemon." in result
        assert "## work" in result

    def test_domain_filter(
        self, stack: MemoryStack, config: MnemonConfig
    ) -> None:
        config.identity_path.write_text("identity", encoding="utf-8")
        config.l1_cache_path.write_text(
            "## work\n- task\n## personal\n- hobby", encoding="utf-8"
        )

        result = stack.wake_up(domain="work")
        assert "## work" in result
        assert "- task" in result
        assert "personal" not in result


# ── status ──────────────────────────────────────────────────────


class TestStatus:
    def test_includes_protocol(
        self, stack: MemoryStack, config: MnemonConfig
    ) -> None:
        config.identity_path.write_text("I am Mnemon.", encoding="utf-8")
        result = stack.status()
        assert PROTOCOL in result
        assert "I am Mnemon." in result
