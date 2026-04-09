"""Canonical JSON store for parsed sessions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from mnemon.parsers import ParsedSession

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_session_id(session_id: str) -> bool:
    """Return True if *session_id* matches the allowed pattern."""
    return bool(SESSION_ID_PATTERN.match(session_id))


def write_canonical(session: ParsedSession, canonical_dir: Path) -> Path:
    """Serialise *session* to canonical JSON and write it to *canonical_dir*.

    Raises ``ValueError`` if the session ID is invalid.
    The write is idempotent — an existing file is silently overwritten.
    Returns the path of the written file.
    """
    if not validate_session_id(session.session_id):
        raise ValueError(
            f"Invalid session_id: {session.session_id!r} "
            f"(must match {SESSION_ID_PATTERN.pattern})"
        )

    canonical_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": 1,
        **asdict(session),
    }

    out_path = canonical_dir / f"{session.session_id}.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


def read_canonical(session_id: str, canonical_dir: Path) -> dict | None:
    """Read and return the canonical JSON for *session_id*, or ``None`` if missing."""
    path = canonical_dir / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_canonical(canonical_dir: Path) -> list[str]:
    """Return session IDs present in *canonical_dir* (sorted)."""
    if not canonical_dir.is_dir():
        return []
    return sorted(p.stem for p in canonical_dir.glob("*.json"))
