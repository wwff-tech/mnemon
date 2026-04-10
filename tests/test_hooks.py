"""Tests for mnemon.hooks — validation, recency scoring, and hardening."""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from mnemon.hooks import (
    _recency_importance,
    main,
    pre_compact_hook,
    save_hook,
    validate_session_id,
)

# ---------------------------------------------------------------------------
# Session ID validation
# ---------------------------------------------------------------------------


class TestValidateSessionId:
    def test_valid_alphanumeric(self) -> None:
        assert validate_session_id("abc123") is True

    def test_valid_with_hyphens_and_underscores(self) -> None:
        assert validate_session_id("session-id_01") is True

    def test_valid_single_char(self) -> None:
        assert validate_session_id("a") is True

    def test_valid_max_length(self) -> None:
        assert validate_session_id("a" * 64) is True

    def test_invalid_empty(self) -> None:
        assert validate_session_id("") is False

    def test_invalid_too_long(self) -> None:
        assert validate_session_id("a" * 65) is False

    def test_invalid_special_chars(self) -> None:
        assert validate_session_id("session;rm -rf /") is False

    def test_invalid_spaces(self) -> None:
        assert validate_session_id("has space") is False

    def test_invalid_dots(self) -> None:
        assert validate_session_id("../etc/passwd") is False


# ---------------------------------------------------------------------------
# Recency importance scoring
# ---------------------------------------------------------------------------


class TestRecencyImportance:
    def test_single_exchange_gets_cap(self) -> None:
        assert _recency_importance(0, 1) == 0.8

    def test_first_of_many_gets_low(self) -> None:
        score = _recency_importance(0, 10)
        assert score == 0.3

    def test_last_of_many_gets_cap(self) -> None:
        score = _recency_importance(9, 10)
        assert score == 0.8

    def test_middle_is_between(self) -> None:
        score = _recency_importance(5, 10)
        assert 0.3 < score < 0.8

    def test_monotonically_increasing(self) -> None:
        scores = [_recency_importance(i, 5) for i in range(5)]
        assert scores == sorted(scores)

    def test_never_exceeds_cap(self) -> None:
        for total in range(1, 20):
            for i in range(total):
                assert _recency_importance(i, total) <= 0.8


# ---------------------------------------------------------------------------
# Hook hardening — validation errors
# ---------------------------------------------------------------------------


class TestSaveHookHardening:
    def test_nonexistent_file_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="mnemon.hooks"):
            save_hook("/nonexistent/path/conversation.json", "valid-session")
        assert "Conversation file not found" in caplog.text

    def test_invalid_session_id_returns_early(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="mnemon.hooks"):
            save_hook("/some/path", "invalid session;drop table")
        assert "Invalid session ID" in caplog.text


class TestPreCompactHookHardening:
    def test_nonexistent_file_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="mnemon.hooks"):
            pre_compact_hook("/nonexistent/path/conversation.json", "valid-session")
        assert "Conversation file not found" in caplog.text

    def test_invalid_session_id_returns_early(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="mnemon.hooks"):
            pre_compact_hook("/some/path", "../traversal")
        assert "Invalid session ID" in caplog.text


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMainArgv:
    """Test argv mode (manual invocation)."""

    def test_two_args_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["hooks", "stop"])
        with pytest.raises(SystemExit, match="1"):
            main()

    def test_too_many_args_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["hooks", "a", "b", "c", "d"])
        with pytest.raises(SystemExit, match="1"):
            main()

    def test_unknown_event_type_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv", ["hooks", "unknown_event", "/some/path", "session1"]
        )
        with pytest.raises(SystemExit, match="1"):
            main()


# ---------------------------------------------------------------------------
# Subprocess invocation hardening
# ---------------------------------------------------------------------------


class TestSubprocessArgvMode:
    """Test argv mode when invoked as a subprocess."""

    def test_wrong_arg_count_returns_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mnemon.hooks", "stop", "only-two"],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 1

    def test_unknown_event_returns_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mnemon.hooks", "bogus", "/tmp/x", "ses1"],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 1

    def test_missing_file_returns_zero(self) -> None:
        """Hooks must never block Claude Code — even on error, exit 0."""
        result = subprocess.run(
            [
                sys.executable, "-m", "mnemon.hooks",
                "stop", "/nonexistent/file.jsonl", "test-session",
            ],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_session_id_returns_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "mnemon.hooks",
                "stop", "/tmp/x", "../../etc/passwd",
            ],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0


class TestSubprocessStdinMode:
    """Test stdin mode (Claude Code hook invocation)."""

    def _run_stdin(self, payload: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-m", "mnemon.hooks"],
            input=payload.encode(),
            capture_output=True,
            timeout=10,
        )

    def test_stop_event_via_stdin(self) -> None:
        import json
        payload = json.dumps({
            "hook_event_name": "Stop",
            "session_id": "test-session",
            "transcript_path": "/nonexistent/file.jsonl",
        })
        result = self._run_stdin(payload)
        # Missing file is not a crash — exits 0
        assert result.returncode == 0

    def test_precompact_event_via_stdin(self) -> None:
        import json
        payload = json.dumps({
            "hook_event_name": "PreCompact",
            "session_id": "test-session",
            "transcript_path": "/nonexistent/file.jsonl",
        })
        result = self._run_stdin(payload)
        assert result.returncode == 0

    def test_unknown_event_via_stdin_exits_1(self) -> None:
        import json
        payload = json.dumps({
            "hook_event_name": "UnknownEvent",
            "session_id": "test-session",
            "transcript_path": "/tmp/x",
        })
        result = self._run_stdin(payload)
        assert result.returncode == 1

    def test_empty_stdin_exits_0(self) -> None:
        result = self._run_stdin("")
        assert result.returncode == 0

    def test_malformed_json_exits_0(self) -> None:
        result = self._run_stdin("{not valid json")
        assert result.returncode == 0


class TestSubprocessTimeout:
    def test_subprocess_respects_timeout(self) -> None:
        """Verify subprocess.run timeout works (caller-side enforcement)."""
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                timeout=1,
            )
