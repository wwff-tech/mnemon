"""Claude Code hooks for Mnemon — save and pre-compact."""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("mnemon.hooks")

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
HOOK_TIMEOUT = 30  # seconds


def validate_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_PATTERN.match(session_id))


def save_hook(conversation_path: str, session_id: str) -> None:
    """Save hook — triggered on Stop event.

    Importance: recency-boosted, capped at 0.8.
    """
    if not validate_session_id(session_id):
        logger.error("Invalid session ID: %s", session_id)
        return

    path = Path(conversation_path)
    if not path.exists():
        logger.error("Conversation file not found: %s", conversation_path)
        return

    try:
        from mnemon.api import Memory
        mem = Memory()
        try:
            mem.ingest(str(path), importance=0.8)
            # Log the save
            log_path = mem._config.saves_log_path
            with open(log_path, "a") as f:
                ts = datetime.now(timezone.utc).isoformat()
                f.write(f"{ts} save session={session_id} path={conversation_path}\n")
        finally:
            mem.close()
    except Exception:
        logger.exception("Save hook failed for session %s", session_id)


def pre_compact_hook(conversation_path: str, session_id: str) -> None:
    """Pre-compact hook — triggered on PreCompact event.

    Same as save hook but with importance floor of 0.7.
    """
    if not validate_session_id(session_id):
        logger.error("Invalid session ID: %s", session_id)
        return

    path = Path(conversation_path)
    if not path.exists():
        logger.error("Conversation file not found: %s", conversation_path)
        return

    try:
        from mnemon.api import Memory
        mem = Memory()
        try:
            mem.ingest(str(path), importance=0.7)
            log_path = mem._config.saves_log_path
            with open(log_path, "a") as f:
                ts = datetime.now(timezone.utc).isoformat()
                f.write(f"{ts} pre_compact session={session_id} path={conversation_path}\n")
        finally:
            mem.close()
    except Exception:
        logger.exception("Pre-compact hook failed for session %s", session_id)


def main() -> None:
    """Entry point for hook subprocess invocation.

    Usage: python -m mnemon.hooks <event_type> <conversation_path> <session_id>

    Hardened: argument list (no shell=True), validated session ID,
    30-second timeout enforced by caller, all exceptions caught.
    """
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <event_type> <conversation_path> <session_id>",
            file=sys.stderr,
        )
        sys.exit(1)

    event_type = sys.argv[1]
    conversation_path = sys.argv[2]
    session_id = sys.argv[3]

    handlers = {
        "stop": save_hook,
        "pre_compact": pre_compact_hook,
    }

    handler = handlers.get(event_type)
    if handler is None:
        logger.error("Unknown event type: %s", event_type)
        sys.exit(1)

    try:
        handler(conversation_path, session_id)
    except Exception:
        logger.exception("Hook failed")
        # Never block Claude Code — swallow and exit cleanly
        sys.exit(0)


if __name__ == "__main__":
    main()
