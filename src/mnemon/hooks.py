"""Claude Code hooks for Mnemon — save and pre-compact.

Invoked as a subprocess by Claude Code on Stop and PreCompact events.
All exceptions are caught and logged — hooks must never block the editor.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("mnemon.hooks")

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
HOOK_TIMEOUT = 30  # seconds — enforced by the calling subprocess

# Importance caps per PRD
_SAVE_IMPORTANCE_CAP = 0.8
_COMPACT_IMPORTANCE_FLOOR = 0.7


def validate_session_id(session_id: str) -> bool:
    """Return True if *session_id* matches the allowed pattern."""
    return bool(SESSION_ID_PATTERN.match(session_id))


def _recency_importance(exchange_index: int, total_exchanges: int) -> float:
    """Compute recency-boosted importance for an exchange.

    More recent exchanges score higher.  The final exchange gets
    ``_SAVE_IMPORTANCE_CAP`` (0.8); the first gets 0.3.  This
    preserves headroom above 0.8 for manually-flagged items.
    """
    if total_exchanges <= 1:
        return _SAVE_IMPORTANCE_CAP
    ratio = exchange_index / (total_exchanges - 1)  # 0.0 → 1.0
    return round(0.3 + ratio * (_SAVE_IMPORTANCE_CAP - 0.3), 3)


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
        from mnemon.parsers import parse

        mem = Memory()
        try:
            # Parse to get exchange count for recency scoring
            session = parse(path)
            n = len(session.exchanges)

            # Compute average recency importance across exchanges
            if n > 0:
                importance = sum(
                    _recency_importance(i, n) for i in range(n)
                ) / n
            else:
                importance = 0.5

            mem.ingest(str(path), importance=min(importance, _SAVE_IMPORTANCE_CAP))

            _log_event(mem, "save", session_id, conversation_path)
        finally:
            mem.close()
    except Exception:
        logger.exception("Save hook failed for session %s", session_id)


def pre_compact_hook(conversation_path: str, session_id: str) -> None:
    """Pre-compact hook — triggered on PreCompact event.

    Same as save hook but with importance floor of 0.7 — if it's being
    compacted, something substantive happened.
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
        from mnemon.parsers import parse

        mem = Memory()
        try:
            session = parse(path)
            n = len(session.exchanges)

            if n > 0:
                importance = sum(
                    _recency_importance(i, n) for i in range(n)
                ) / n
            else:
                importance = _COMPACT_IMPORTANCE_FLOOR

            # Apply the compact floor
            importance = max(importance, _COMPACT_IMPORTANCE_FLOOR)

            mem.ingest(str(path), importance=importance)

            _log_event(mem, "pre_compact", session_id, conversation_path)
        finally:
            mem.close()
    except Exception:
        logger.exception("Pre-compact hook failed for session %s", session_id)


def _log_event(
    mem: object, event: str, session_id: str, conversation_path: str
) -> None:
    """Append a line to the saves log."""
    from mnemon.api import Memory

    assert isinstance(mem, Memory)
    log_path = mem._config.saves_log_path
    with open(log_path, "a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts} {event} session={session_id} path={conversation_path}\n")


def main() -> None:
    """Entry point for hook subprocess invocation.

    Supports two invocation modes:

    **stdin mode** (Claude Code hooks — preferred)::

        echo '{"hook_event_name":"Stop","session_id":"abc","transcript_path":"/p"}' \\
            | python -m mnemon.hooks

    Claude Code passes a JSON object on stdin with ``hook_event_name``,
    ``session_id``, and ``transcript_path``.

    **argv mode** (manual/testing)::

        python -m mnemon.hooks <event_type> <conversation_path> <session_id>

    Hardened: no shell=True, validated session IDs, all exceptions caught.
    """
    event_type: str | None = None
    conversation_path: str | None = None
    session_id: str | None = None

    if len(sys.argv) == 4:
        # argv mode
        event_type = sys.argv[1]
        conversation_path = sys.argv[2]
        session_id = sys.argv[3]
    elif len(sys.argv) == 1:
        # stdin mode — read JSON from Claude Code
        try:
            import json

            raw = sys.stdin.read()
            if not raw.strip():
                logger.error("Empty stdin — nothing to process")
                sys.exit(0)
            data = json.loads(raw)
            event_type = data.get("hook_event_name", "")
            session_id = data.get("session_id", "")
            conversation_path = data.get("transcript_path", "")
        except Exception:
            logger.exception("Failed to parse stdin JSON")
            sys.exit(0)
    else:
        print(
            f"Usage: {sys.argv[0]} [<event_type> <conversation_path> <session_id>]",
            file=sys.stderr,
        )
        print("  Or pipe Claude Code hook JSON to stdin.", file=sys.stderr)
        sys.exit(1)

    # Normalise event names — Claude Code sends "Stop" / "PreCompact",
    # our handlers use lowercase.
    event_map: dict[str, str] = {
        "stop": "stop",
        "precompact": "pre_compact",
        "pre_compact": "pre_compact",
    }
    normalised = event_map.get((event_type or "").lower().replace("-", ""))

    handlers = {
        "stop": save_hook,
        "pre_compact": pre_compact_hook,
    }

    handler = handlers.get(normalised or "")
    if handler is None:
        logger.error("Unknown event type: %s", event_type)
        sys.exit(1)

    try:
        handler(conversation_path or "", session_id or "")
    except Exception:
        logger.exception("Hook failed")
        # Never block Claude Code — swallow and exit cleanly
        sys.exit(0)


if __name__ == "__main__":
    main()
