"""Tests for the canonical store module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemon.canonical import (
    list_canonical,
    read_canonical,
    validate_session_id,
    write_canonical,
)
from mnemon.parsers import ParsedSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(**overrides: object) -> ParsedSession:
    defaults = {
        "session_id": "session_abc",
        "source_format": "claude_jsonl",
        "source_path": "/tmp/session_abc.jsonl",
        "first_event_at": "2026-01-15T10:00:00Z",
        "last_event_at": "2026-01-15T11:23:00Z",
        "exchanges": [
            {
                "exchange_index": 0,
                "user": "What is the auth flow?",
                "assistant": "The auth flow uses JWT tokens...",
                "timestamp": "2026-01-15T10:01:00Z",
            }
        ],
    }
    defaults.update(overrides)
    return ParsedSession(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Session ID validation
# ---------------------------------------------------------------------------


class TestValidateSessionId:
    def test_valid_alphanumeric(self) -> None:
        assert validate_session_id("abc123") is True

    def test_valid_with_hyphens_underscores(self) -> None:
        assert validate_session_id("my_session-01") is True

    def test_valid_single_char(self) -> None:
        assert validate_session_id("a") is True

    def test_valid_max_length(self) -> None:
        assert validate_session_id("a" * 64) is True

    def test_invalid_empty(self) -> None:
        assert validate_session_id("") is False

    def test_invalid_too_long(self) -> None:
        assert validate_session_id("a" * 65) is False

    def test_invalid_special_chars(self) -> None:
        assert validate_session_id("bad/id") is False
        assert validate_session_id("bad id") is False
        assert validate_session_id("bad.id") is False


# ---------------------------------------------------------------------------
# Write / read round-trip
# ---------------------------------------------------------------------------


class TestWriteAndRead:
    def test_round_trip(self, tmp_path: Path) -> None:
        session = _make_session()
        out = write_canonical(session, tmp_path)

        assert out == tmp_path / "session_abc.json"
        assert out.exists()

        data = read_canonical("session_abc", tmp_path)
        assert data is not None
        assert data["session_id"] == "session_abc"
        assert data["source_format"] == "claude_jsonl"
        assert data["first_event_at"] == "2026-01-15T10:00:00Z"
        assert data["last_event_at"] == "2026-01-15T11:23:00Z"
        assert len(data["exchanges"]) == 1
        assert data["exchanges"][0]["user"] == "What is the auth flow?"

    def test_schema_version_present(self, tmp_path: Path) -> None:
        session = _make_session()
        write_canonical(session, tmp_path)
        data = read_canonical("session_abc", tmp_path)
        assert data is not None
        assert data["schema_version"] == 1

    def test_idempotent_overwrite(self, tmp_path: Path) -> None:
        session = _make_session()
        p1 = write_canonical(session, tmp_path)

        # Write again with different content
        session2 = _make_session(source_path="/tmp/other.jsonl")
        p2 = write_canonical(session2, tmp_path)

        assert p1 == p2
        data = read_canonical("session_abc", tmp_path)
        assert data is not None
        assert data["source_path"] == "/tmp/other.jsonl"

    def test_invalid_session_id_raises(self, tmp_path: Path) -> None:
        session = _make_session(session_id="bad/id")
        with pytest.raises(ValueError, match="Invalid session_id"):
            write_canonical(session, tmp_path)


# ---------------------------------------------------------------------------
# read_canonical
# ---------------------------------------------------------------------------


class TestReadCanonical:
    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        assert read_canonical("nonexistent", tmp_path) is None


# ---------------------------------------------------------------------------
# list_canonical
# ---------------------------------------------------------------------------


class TestListCanonical:
    def test_lists_session_ids(self, tmp_path: Path) -> None:
        write_canonical(_make_session(session_id="aaa"), tmp_path)
        write_canonical(_make_session(session_id="zzz"), tmp_path)
        write_canonical(_make_session(session_id="mmm"), tmp_path)

        ids = list_canonical(tmp_path)
        assert ids == ["aaa", "mmm", "zzz"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert list_canonical(tmp_path) == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert list_canonical(tmp_path / "nope") == []
