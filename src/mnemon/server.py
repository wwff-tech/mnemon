"""MCP server exposing Mnemon memory tools via fastmcp."""

from __future__ import annotations

import json
import logging

from fastmcp import FastMCP
from starlette.middleware import Middleware

from mnemon.api import Memory
from mnemon.auth import AuthMiddleware, log_auth_startup
from mnemon.config import load_config

# ---------------------------------------------------------------------------
# Module-level Memory instance, lazily initialised
# ---------------------------------------------------------------------------

_memory: Memory | None = None


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("mnemon")


# 1 ── status ───────────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_status() -> str:
    """Return agent identity and critical facts (L0 + L1) with search-first protocol."""
    return get_memory().status()


# 2 ── wake_up ──────────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_wake_up(domain: str | None = None) -> str:
    """Load identity and top-of-mind context, optionally scoped to a domain."""
    return get_memory().wake_up(domain=domain)


# 3 ── search ───────────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_search(
    query: str,
    domain: str | None = None,
    topic: str | None = None,
    n: int = 5,
) -> str:
    """Search memory for relevant chunks. Returns JSON list of results."""
    results = get_memory().search(query, domain=domain, topic=topic, n=n)
    return json.dumps(
        [
            {
                "id": r.id,
                "text": r.text,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in results
        ],
        default=str,
    )


# 4 ── list_domains ────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_list_domains() -> str:
    """List all memory domains with document counts. Returns JSON."""
    return json.dumps(get_memory().list_domains())


# 5 ── list_topics ─────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_list_topics(domain: str) -> str:
    """List topics within a domain with document counts. Returns JSON."""
    return json.dumps(get_memory().list_topics(domain))


# 6 ── kg_query ─────────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_kg_query(entity: str, as_of: str | None = None) -> str:
    """Query the knowledge graph for facts about an entity. Returns JSON."""
    return json.dumps(get_memory().kg.query(entity, as_of=as_of))


# 7 ── kg_timeline ─────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_kg_timeline(entity: str) -> str:
    """Return the full timeline of knowledge-graph facts for an entity. Returns JSON."""
    return json.dumps(get_memory().kg.timeline(entity))


# 8 ── add ──────────────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_add(
    text: str,
    domain: str,
    topic: str | None = None,
    importance: float = 0.5,
    source: str | None = None,
) -> str:
    """Add a text chunk to memory. Returns confirmation with chunk ID."""
    chunk_id = get_memory().add(
        text, domain=domain, topic=topic, importance=importance, source=source
    )
    return json.dumps({"ok": True, "chunk_id": chunk_id})


# 9 ── kg_add ──────────────────────────────────────────────────────────────
@mcp.tool()
def mnemon_kg_add(
    subject: str,
    predicate: str,
    object: str,
    valid_from: str,
    subject_type: str = "concept",
    object_type: str = "concept",
) -> str:
    """Add a knowledge-graph triple. Returns confirmation with edge ID."""
    edge_id = get_memory().kg.add(
        subject,
        predicate,
        object,
        valid_from=valid_from,
        subject_type=subject_type,
        object_type=object_type,
    )
    return json.dumps({"ok": True, "edge_id": edge_id})


# 10 ── kg_invalidate ──────────────────────────────────────────────────────
@mcp.tool()
def mnemon_kg_invalidate(
    subject: str,
    predicate: str,
    object: str,
    ended: str,
) -> str:
    """Mark a knowledge-graph triple as no longer valid."""
    success = get_memory().kg.invalidate(subject, predicate, object, ended=ended)
    if success:
        return json.dumps({"ok": True, "message": "Triple invalidated."})
    return json.dumps({"ok": False, "message": "Triple not found or already invalidated."})


# 11 ── check_duplicate ────────────────────────────────────────────────────
@mcp.tool()
def mnemon_check_duplicate(text: str) -> str:
    """Check whether a text chunk has already been ingested. Returns JSON."""
    is_dup = get_memory().check_duplicate(text)
    return json.dumps({"is_duplicate": is_dup})


# 12 ── review_list ────────────────────────────────────────────────────
@mcp.tool()
def mnemon_review_list(limit: int = 20) -> str:
    """List unresolved low-confidence domain assignments for human review. Returns JSON."""
    items = get_memory().review.list(limit=limit)
    return json.dumps(items, default=str)


# 13 ── review_resolve ─────────────────────────────────────────────────
@mcp.tool()
def mnemon_review_resolve(
    chunk_id: str,
    domain: str,
    topic: str | None = None,
) -> str:
    """Confirm or correct a low-confidence domain assignment."""
    get_memory().review.resolve(chunk_id=chunk_id, domain=domain, topic=topic)
    return json.dumps({"ok": True, "chunk_id": chunk_id, "domain": domain, "topic": topic})


# 14 ── backup_prep ───────────────────────────────────────────────────────
@mcp.tool()
def mnemon_backup_prep() -> str:
    """Quiesce writes and prepare for backup. Returns lock file path."""
    lock_path = get_memory().backup_prep()
    return json.dumps({"ok": True, "lock_path": lock_path})


# 15 ── backup_release ────────────────────────────────────────────────────
@mcp.tool()
def mnemon_backup_release() -> str:
    """Release backup lock and resume normal operation."""
    get_memory().backup_release()
    return json.dumps({"ok": True, "message": "Backup lock released."})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    log_auth_startup(config.auth_mode)

    middleware = [
        Middleware(
            AuthMiddleware,
            auth_mode=config.auth_mode,
            auth_token=config.auth_token,
        ),
    ]

    app = mcp.http_app(transport="streamable-http", middleware=middleware)

    import uvicorn

    uvicorn.run(app, host=config.bind_host, port=config.bind_port)


if __name__ == "__main__":
    main()
