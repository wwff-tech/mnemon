"""Tests for the vector store abstraction."""

from __future__ import annotations

import pytest

from mnemon.vectorstore import ChromaStore


@pytest.fixture()
def store(tmp_path):
    """Create a ChromaStore backed by a temp directory."""
    return ChromaStore(persist_directory=tmp_path / "chroma", collection_name="test")


class TestAddSearchDeleteGetRoundTrip:
    def test_add_and_get(self, store):
        store.add("doc1", "hello world", {"topic": "greeting"})
        result = store.get("doc1")
        assert result is not None
        assert result.id == "doc1"
        assert result.text == "hello world"
        assert result.metadata == {"topic": "greeting"}

    def test_add_search_returns_result(self, store):
        store.add("doc1", "the cat sat on the mat", {"topic": "animals"})
        results = store.search("cat on a mat")
        assert len(results) >= 1
        assert results[0].id == "doc1"
        assert isinstance(results[0].score, float)

    def test_delete_removes_document(self, store):
        store.add("doc1", "some text", {"topic": "test"})
        assert store.get("doc1") is not None
        store.delete("doc1")
        assert store.get("doc1") is None


class TestWhereFiltering:
    def test_search_with_where(self, store):
        store.add("a", "python programming language", {"lang": "python"})
        store.add("b", "java programming language", {"lang": "java"})
        results = store.search("programming", n=5, where={"lang": "python"})
        assert len(results) == 1
        assert results[0].id == "a"

    def test_search_where_no_match(self, store):
        store.add("a", "hello world", {"lang": "python"})
        results = store.search("hello", where={"lang": "rust"})
        assert results == []


class TestListMetadata:
    def test_returns_all_metadata(self, store):
        store.add("a", "text a", {"tag": "one"})
        store.add("b", "text b", {"tag": "two"})
        metas = store.list_metadata()
        tags = sorted(m["tag"] for m in metas)
        assert tags == ["one", "two"]

    def test_filtered_by_where(self, store):
        store.add("a", "text a", {"tag": "one"})
        store.add("b", "text b", {"tag": "two"})
        metas = store.list_metadata(where={"tag": "one"})
        assert len(metas) == 1
        assert metas[0]["tag"] == "one"


class TestCount:
    def test_empty_store(self, store):
        assert store.count() == 0

    def test_tracks_adds(self, store):
        store.add("a", "text a", {"x": "1"})
        assert store.count() == 1
        store.add("b", "text b", {"x": "2"})
        assert store.count() == 2

    def test_tracks_deletes(self, store):
        store.add("a", "text a", {"x": "1"})
        store.add("b", "text b", {"x": "2"})
        store.delete("a")
        assert store.count() == 1


class TestSearchSorting:
    def test_results_sorted_by_score_descending(self, store):
        store.add("close", "the cat sat on the mat", {"x": "1"})
        store.add("far", "quantum mechanics and thermodynamics", {"x": "2"})
        results = store.search("cat mat", n=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score


class TestGetMissing:
    def test_get_returns_none_for_missing_id(self, store):
        assert store.get("nonexistent") is None
