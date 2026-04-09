"""Parsers for Claude session export formats."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedSession:
    session_id: str
    source_format: str
    source_path: str
    first_event_at: str | None = None
    last_event_at: str | None = None
    exchanges: list[dict[str, Any]] = field(default_factory=list)


def _extract_text(content: str | list[Any]) -> str:
    """Extract plain text from a content field that may be a string or list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _session_id_from_path(path: Path) -> str:
    """Derive a session ID from filename stem, falling back to a hash."""
    stem = path.stem
    if stem:
        return stem
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


def parse_claude_jsonl(path: str | Path) -> ParsedSession:
    """Parse a Claude Code JSONL session file into a ParsedSession."""
    path = Path(path)
    messages: list[dict[str, Any]] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = obj.get("type")
            if msg_type in ("human", "assistant"):
                messages.append(obj)

    exchanges: list[dict[str, Any]] = []
    timestamps: list[str] = []
    idx = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("type") == "human":
            user_text = _extract_text(msg.get("content", ""))
            ts = msg.get("timestamp") or msg.get("created_at")
            if ts:
                timestamps.append(ts)

            # Look for a paired assistant message
            assistant_text = ""
            assistant_ts = None
            if i + 1 < len(messages) and messages[i + 1].get("type") == "assistant":
                assistant_msg = messages[i + 1]
                assistant_text = _extract_text(assistant_msg.get("content", ""))
                assistant_ts = assistant_msg.get("timestamp") or assistant_msg.get("created_at")
                if assistant_ts:
                    timestamps.append(assistant_ts)
                i += 2
            else:
                i += 1

            exchanges.append({
                "exchange_index": idx,
                "user": user_text,
                "assistant": assistant_text,
                "timestamp": ts,
            })
            idx += 1
        else:
            # Skip unpaired assistant messages
            i += 1

    timestamps.sort()

    return ParsedSession(
        session_id=_session_id_from_path(path),
        source_format="claude_jsonl",
        source_path=str(path),
        first_event_at=timestamps[0] if timestamps else None,
        last_event_at=timestamps[-1] if timestamps else None,
        exchanges=exchanges,
    )


def parse_claude_ai_json(path: str | Path) -> ParsedSession:
    """Parse a Claude.ai JSON conversation export into a ParsedSession."""
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    # Determine session ID
    session_id = data.get("uuid") or _session_id_from_path(path)

    # Find the messages array
    messages: list[dict[str, Any]] = []
    if isinstance(data.get("chat_messages"), list):
        messages = data["chat_messages"]
    elif isinstance(data.get("messages"), list):
        messages = data["messages"]

    exchanges: list[dict[str, Any]] = []
    timestamps: list[str] = []
    idx = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        sender = msg.get("sender", "")
        if sender == "human":
            user_text = msg.get("text") or _extract_text(msg.get("content", ""))
            ts = msg.get("created_at") or msg.get("timestamp")
            if ts:
                timestamps.append(ts)

            assistant_text = ""
            assistant_ts = None
            if i + 1 < len(messages) and messages[i + 1].get("sender") == "assistant":
                a_msg = messages[i + 1]
                assistant_text = a_msg.get("text") or _extract_text(a_msg.get("content", ""))
                assistant_ts = a_msg.get("created_at") or a_msg.get("timestamp")
                if assistant_ts:
                    timestamps.append(assistant_ts)
                i += 2
            else:
                i += 1

            exchanges.append({
                "exchange_index": idx,
                "user": user_text,
                "assistant": assistant_text,
                "timestamp": ts,
            })
            idx += 1
        else:
            i += 1

    timestamps.sort()

    return ParsedSession(
        session_id=session_id,
        source_format="claude_ai_json",
        source_path=str(path),
        first_event_at=timestamps[0] if timestamps else None,
        last_event_at=timestamps[-1] if timestamps else None,
        exchanges=exchanges,
    )


def detect_format(path: str | Path) -> str:
    """Detect the session export format from file extension and structure."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return "claude_jsonl"

    if suffix == ".json":
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Cannot read JSON file: {path}") from exc

        if isinstance(data, dict) and (
            "chat_messages" in data or "messages" in data or "uuid" in data
        ):
            return "claude_ai_json"

        raise ValueError(f"Unrecognised JSON structure in {path}")

    raise ValueError(f"Unknown file extension: {suffix}")


def parse(path: str | Path) -> ParsedSession:
    """Auto-detect format and parse a session file."""
    fmt = detect_format(path)
    if fmt == "claude_jsonl":
        return parse_claude_jsonl(path)
    if fmt == "claude_ai_json":
        return parse_claude_ai_json(path)
    raise ValueError(f"No parser for format: {fmt}")
