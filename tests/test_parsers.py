"""Tests for the parsers module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemon.parsers import (
    ParsedSession,
    detect_format,
    parse,
    parse_claude_ai_json,
    parse_claude_jsonl,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

JSONL_LINES = [
    {"type": "human", "content": "What is the auth flow?", "timestamp": "2026-01-15T10:00:00Z"},
    {
        "type": "assistant",
        "content": "The auth flow uses JWT tokens...",
        "timestamp": "2026-01-15T10:00:05Z",
    },
    {
        "type": "human",
        "content": "How do we handle refresh?",
        "timestamp": "2026-01-15T10:01:00Z",
    },
    {
        "type": "assistant",
        "content": "Refresh tokens are stored in httpOnly cookies...",
        "timestamp": "2026-01-15T10:01:10Z",
    },
]

CLAUDE_AI_JSON = {
    "uuid": "conv-123",
    "name": "Auth Discussion",
    "chat_messages": [
        {
            "sender": "human",
            "text": "What is the auth flow?",
            "created_at": "2026-01-15T10:00:00Z",
        },
        {
            "sender": "assistant",
            "text": "The auth flow uses JWT tokens...",
            "created_at": "2026-01-15T10:00:05Z",
        },
    ],
}


@pytest.fixture
def jsonl_file(tmp_path: Path) -> Path:
    p = tmp_path / "session_abc.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in JSONL_LINES) + "\n")
    return p


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    p = tmp_path / "conv-123.json"
    p.write_text(json.dumps(CLAUDE_AI_JSON))
    return p


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


class TestParseClaudeJsonl:
    def test_exchange_count(self, jsonl_file: Path) -> None:
        result = parse_claude_jsonl(jsonl_file)
        assert len(result.exchanges) == 2

    def test_exchange_pairing(self, jsonl_file: Path) -> None:
        result = parse_claude_jsonl(jsonl_file)
        ex0 = result.exchanges[0]
        assert ex0["user"] == "What is the auth flow?"
        assert ex0["assistant"] == "The auth flow uses JWT tokens..."
        assert ex0["exchange_index"] == 0

        ex1 = result.exchanges[1]
        assert ex1["user"] == "How do we handle refresh?"
        assert ex1["assistant"] == "Refresh tokens are stored in httpOnly cookies..."
        assert ex1["exchange_index"] == 1

    def test_timestamps_extracted(self, jsonl_file: Path) -> None:
        result = parse_claude_jsonl(jsonl_file)
        assert result.first_event_at == "2026-01-15T10:00:00Z"
        assert result.last_event_at == "2026-01-15T10:01:10Z"
        assert result.exchanges[0]["timestamp"] == "2026-01-15T10:00:00Z"

    def test_session_id_from_stem(self, jsonl_file: Path) -> None:
        result = parse_claude_jsonl(jsonl_file)
        assert result.session_id == "session_abc"

    def test_source_format(self, jsonl_file: Path) -> None:
        result = parse_claude_jsonl(jsonl_file)
        assert result.source_format == "claude_jsonl"

    def test_content_as_list_of_blocks(self, tmp_path: Path) -> None:
        """Content may be a list of content blocks instead of a plain string."""
        lines = [
            {
                "type": "human",
                "content": [{"type": "text", "text": "Hello"}],
                "timestamp": "2026-01-15T10:00:00Z",
            },
            {
                "type": "assistant",
                "content": [
                    {"type": "text", "text": "Hi there!"},
                    {"type": "text", "text": "How can I help?"},
                ],
                "timestamp": "2026-01-15T10:00:05Z",
            },
        ]
        p = tmp_path / "blocks.jsonl"
        p.write_text("\n".join(json.dumps(entry) for entry in lines) + "\n")
        result = parse_claude_jsonl(p)
        assert len(result.exchanges) == 1
        assert result.exchanges[0]["user"] == "Hello"
        assert "Hi there!" in result.exchanges[0]["assistant"]
        assert "How can I help?" in result.exchanges[0]["assistant"]

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON lines are silently skipped."""
        content = (
            '{"type": "human", "content": "Hi", "timestamp": "2026-01-15T10:00:00Z"}\n'
            "NOT VALID JSON\n"
            '{"type": "assistant", "content": "Hello!", "timestamp": "2026-01-15T10:00:01Z"}\n'
        )
        p = tmp_path / "bad.jsonl"
        p.write_text(content)
        result = parse_claude_jsonl(p)
        assert len(result.exchanges) == 1
        assert result.exchanges[0]["user"] == "Hi"

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = parse_claude_jsonl(p)
        assert result.exchanges == []
        assert result.first_event_at is None


# ---------------------------------------------------------------------------
# Claude.ai JSON parsing
# ---------------------------------------------------------------------------


class TestParseClaudeAiJson:
    def test_exchange_count(self, json_file: Path) -> None:
        result = parse_claude_ai_json(json_file)
        assert len(result.exchanges) == 1

    def test_exchange_pairing(self, json_file: Path) -> None:
        result = parse_claude_ai_json(json_file)
        ex = result.exchanges[0]
        assert ex["user"] == "What is the auth flow?"
        assert ex["assistant"] == "The auth flow uses JWT tokens..."

    def test_session_id_from_uuid(self, json_file: Path) -> None:
        result = parse_claude_ai_json(json_file)
        assert result.session_id == "conv-123"

    def test_timestamps(self, json_file: Path) -> None:
        result = parse_claude_ai_json(json_file)
        assert result.first_event_at == "2026-01-15T10:00:00Z"
        assert result.last_event_at == "2026-01-15T10:00:05Z"

    def test_source_format(self, json_file: Path) -> None:
        result = parse_claude_ai_json(json_file)
        assert result.source_format == "claude_ai_json"

    def test_fallback_session_id_without_uuid(self, tmp_path: Path) -> None:
        data = {
            "chat_messages": [
                {"sender": "human", "text": "Hey"},
                {"sender": "assistant", "text": "Hello!"},
            ]
        }
        p = tmp_path / "no_uuid.json"
        p.write_text(json.dumps(data))
        result = parse_claude_ai_json(p)
        assert result.session_id == "no_uuid"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_jsonl_extension(self, jsonl_file: Path) -> None:
        assert detect_format(jsonl_file) == "claude_jsonl"

    def test_json_with_chat_messages(self, json_file: Path) -> None:
        assert detect_format(json_file) == "claude_ai_json"

    def test_json_with_messages_key(self, tmp_path: Path) -> None:
        p = tmp_path / "alt.json"
        p.write_text(json.dumps({"messages": []}))
        assert detect_format(p) == "claude_ai_json"

    def test_unknown_json_structure(self, tmp_path: Path) -> None:
        p = tmp_path / "random.json"
        p.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError, match="Unrecognised JSON structure"):
            detect_format(p)

    def test_unknown_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unknown file extension"):
            detect_format(p)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestParse:
    def test_dispatches_jsonl(self, jsonl_file: Path) -> None:
        result = parse(jsonl_file)
        assert isinstance(result, ParsedSession)
        assert result.source_format == "claude_jsonl"
        assert len(result.exchanges) == 2

    def test_dispatches_json(self, json_file: Path) -> None:
        result = parse(json_file)
        assert isinstance(result, ParsedSession)
        assert result.source_format == "claude_ai_json"
        assert len(result.exchanges) == 1
