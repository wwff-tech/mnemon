"""Tests for mnemon.chunking."""

from __future__ import annotations

import hashlib

from mnemon.chunking import (
    chunk_exchanges,
    chunk_paragraphs,
    compute_chunk_hash,
    estimate_tokens,
    sub_chunk,
)

# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_known_lengths(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("a") == 1  # 1 char → ceil(1/5) = 1
        assert estimate_tokens("abcde") == 1  # exactly 5 chars
        assert estimate_tokens("abcdef") == 2  # 6 chars → 1 + 1
        assert estimate_tokens("a" * 900) == 180  # PRD baseline

    def test_single_word(self) -> None:
        assert estimate_tokens("hello") == 1


# ---------------------------------------------------------------------------
# compute_chunk_hash
# ---------------------------------------------------------------------------


class TestComputeChunkHash:
    def test_sha256(self) -> None:
        text = "hello world"
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert compute_chunk_hash(text) == expected

    def test_empty(self) -> None:
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_chunk_hash("") == expected


# ---------------------------------------------------------------------------
# sub_chunk
# ---------------------------------------------------------------------------


class TestSubChunk:
    def test_short_text_no_split(self) -> None:
        segments = sub_chunk("short", max_tokens=10)
        assert segments == ["short"]

    def test_overlap_correct(self) -> None:
        # 20 tokens → 100 chars max per segment, overlap 20
        text = "A" * 200  # needs splitting
        segments = sub_chunk(text, max_tokens=20, overlap_chars=20)
        assert len(segments) > 1
        # Each segment is at most 100 chars
        for seg in segments:
            assert len(seg) <= 100
        # Verify overlap: end of seg[0] == start of seg[1]
        overlap = 20
        assert segments[0][-overlap:] == segments[1][:overlap]

    def test_empty_text(self) -> None:
        assert sub_chunk("") == []

    def test_exact_boundary(self) -> None:
        text = "X" * 100
        segments = sub_chunk(text, max_tokens=20, overlap_chars=0)
        assert segments == [text]  # exactly at max_chars


# ---------------------------------------------------------------------------
# chunk_exchanges
# ---------------------------------------------------------------------------


class TestChunkExchanges:
    def test_short_exchange_single_chunk(self) -> None:
        exchanges = [
            {
                "exchange_index": 0,
                "user": "Hi",
                "assistant": "Hello!",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ]
        chunks = chunk_exchanges(exchanges)
        assert len(chunks) == 1
        assert chunks[0].text == "Hi\n\nHello!"
        assert chunks[0].exchange_index == 0
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["timestamp"] == "2025-01-01T00:00:00Z"
        assert chunks[0].content_hash == compute_chunk_hash("Hi\n\nHello!")

    def test_long_exchange_multiple_sub_chunks(self) -> None:
        long_text = "word " * 500  # ~2500 chars → well over 180 tokens
        exchanges = [
            {
                "exchange_index": 3,
                "user": long_text,
                "assistant": long_text,
                "timestamp": "2025-06-15T12:00:00Z",
            }
        ]
        chunks = chunk_exchanges(exchanges, max_tokens=180, overlap_chars=100)
        assert len(chunks) > 1
        # All sub-chunks share the same exchange_index
        for c in chunks:
            assert c.exchange_index == 3
            assert c.metadata["timestamp"] == "2025-06-15T12:00:00Z"
            assert c.content_hash != ""

    def test_chunk_index_sequential(self) -> None:
        exchanges = [
            {"exchange_index": 0, "user": "a", "assistant": "b", "timestamp": "t0"},
            {"exchange_index": 1, "user": "c", "assistant": "d", "timestamp": "t1"},
        ]
        chunks = chunk_exchanges(exchanges)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_exchange_index_preserved_across_sub_chunks(self) -> None:
        long_text = "x" * 2000
        exchanges = [
            {"exchange_index": 7, "user": long_text, "assistant": long_text},
        ]
        chunks = chunk_exchanges(exchanges, max_tokens=50, overlap_chars=20)
        assert len(chunks) > 1
        for c in chunks:
            assert c.exchange_index == 7

    def test_no_timestamp(self) -> None:
        exchanges = [
            {"exchange_index": 0, "user": "a", "assistant": "b"},
        ]
        chunks = chunk_exchanges(exchanges)
        assert "timestamp" not in chunks[0].metadata


# ---------------------------------------------------------------------------
# chunk_paragraphs
# ---------------------------------------------------------------------------


class TestChunkParagraphs:
    def test_merges_short_paragraphs(self) -> None:
        text = "Short one.\n\nShort two.\n\nShort three."
        chunks = chunk_paragraphs(text, max_tokens=200)
        # All three paragraphs are very short; they should merge into one chunk.
        assert len(chunks) == 1
        assert "Short one." in chunks[0].text
        assert "Short three." in chunks[0].text

    def test_hard_splits_long_paragraph(self) -> None:
        # Single paragraph exceeding max_tokens → hard split.
        text = "A" * 2000  # 2000 chars, max_tokens=50 → max 250 chars/chunk
        chunks = chunk_paragraphs(text, max_tokens=50)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 250

    def test_no_exchange_index(self) -> None:
        chunks = chunk_paragraphs("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].exchange_index is None

    def test_empty_text(self) -> None:
        assert chunk_paragraphs("") == []

    def test_content_hash_set(self) -> None:
        chunks = chunk_paragraphs("Some text here.")
        assert chunks[0].content_hash == compute_chunk_hash("Some text here.")

    def test_chunk_index_sequential(self) -> None:
        text = "Para one.\n\n" + "B" * 2000 + "\n\nPara last."
        chunks = chunk_paragraphs(text, max_tokens=50)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
