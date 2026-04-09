"""Tests for mnemon.hooks — validation and hardening logic."""

from __future__ import annotations

import logging

import pytest

from mnemon.hooks import main, save_hook, validate_session_id


class TestValidateSessionId:
    def test_valid_alphanumeric(self):
        assert validate_session_id("abc123") is True

    def test_valid_with_hyphens_and_underscores(self):
        assert validate_session_id("session-id_01") is True

    def test_valid_single_char(self):
        assert validate_session_id("a") is True

    def test_valid_max_length(self):
        assert validate_session_id("a" * 64) is True

    def test_invalid_empty(self):
        assert validate_session_id("") is False

    def test_invalid_too_long(self):
        assert validate_session_id("a" * 65) is False

    def test_invalid_special_chars(self):
        assert validate_session_id("session;rm -rf /") is False

    def test_invalid_spaces(self):
        assert validate_session_id("has space") is False

    def test_invalid_dots(self):
        assert validate_session_id("../etc/passwd") is False


class TestSaveHookHardening:
    def test_nonexistent_file_logs_error(self, caplog):
        with caplog.at_level(logging.ERROR, logger="mnemon.hooks"):
            save_hook("/nonexistent/path/conversation.json", "valid-session")
        assert "Conversation file not found" in caplog.text

    def test_invalid_session_id_returns_early(self, caplog):
        with caplog.at_level(logging.ERROR, logger="mnemon.hooks"):
            save_hook("/some/path", "invalid session;drop table")
        assert "Invalid session ID" in caplog.text


class TestMain:
    def test_wrong_arg_count_exits_1(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["hooks"])
        with pytest.raises(SystemExit, match="1"):
            main()

    def test_too_many_args_exits_1(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["hooks", "a", "b", "c", "d"])
        with pytest.raises(SystemExit, match="1"):
            main()

    def test_unknown_event_type_exits_1(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["hooks", "unknown_event", "/some/path", "session1"])
        with pytest.raises(SystemExit, match="1"):
            main()
