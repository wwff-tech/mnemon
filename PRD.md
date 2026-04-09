# Mnemon — Agent Memory System Pre-PRD

**Status:** v2, working document  
**Author:** Mike Preston  
**Date:** April 2026  
**Codename:** Mnemon  

---

## Problem Statement

Every agent session starts cold. Six months of debugging decisions, architecture debates, project context, and hard-won preferences — gone when the context window resets. The naive solutions are either unaffordable (stuff everything in context, pay per token), lossy (LLM summarisation discards the reasoning you actually needed), or cloud-dependent (Mem0, Zep — subscription fees, data leaving the machine).

MemPalace identified the right shape of the solution, then got distracted by marketing a compression dialect that regresses retrieval performance. This document describes what to actually build, incorporating the refinements from the design review.

---

## Scope

A network-first, API-key-free persistent memory layer for agentic workflows, exposing a Python API and an HTTP MCP server. Primary consumers: Claude Code (via hooks + MCP), Hermes meta-agent. Secondary: any MCP-compatible agent on any machine in the fleet.

This is **not** a general-purpose RAG system. It is purpose-built for conversational and project memory across sessions.

---

## What We Are Stealing From MemPalace

**Hierarchical metadata as a retrieval filter.** The 34% retrieval lift from domain + topic filtering over flat vector search is real — it's disciplined metadata, not a novel algorithm. Commit to the taxonomy at design time, enforce it at ingest.

**The 4-layer memory stack.** Progressive loading is the correct approach to token budget management. L0 (identity, ~100 tokens, always loaded), L1 (top-N important memories pre-computed, ~500–800 tokens, always loaded), L2 (on-demand by topic, ~200–500 tokens), L3 (full semantic search, on demand). Total wake-up cost under 900 tokens.

**Temporal knowledge graph in SQLite.** Entity-relationship triples with `valid_from`/`valid_to` validity windows. Point-in-time queries. The most original useful work in the project — local Graphiti without Neo4j.

**Pre-compact hook.** Structured save triggered before Claude Code context compression. Defensive, necessary.

**Exchange-pair chunking for conversations.** One user turn + one AI response per chunk. Cleaner semantic boundaries than arbitrary character splits for dialogue data. Sub-chunk within an exchange when it exceeds the token budget; tag sub-chunks with shared `exchange_index` so they can be reconstructed from the canonical store.

**"Know before speaking" protocol.** Inject a search-first instruction into the MCP status response. Cheap enforcement of the memory-before-hallucination pattern.

---

## What We Are Not Building

**AAAK or any custom compression dialect.** Regresses benchmark performance, doesn't save tokens at realistic scales, and adds complexity for no gain. Store verbatim text.

**The palace navigation graph.** Tunnels, BFS traversal, fuzzy room matching — the retrieval improvement comes from the metadata filter layer, not graph traversal.

**A bespoke CLI.** Python API and MCP server are the interfaces. If a thin CLI shim falls out naturally (for hook invocation), fine — it is not a deliverable.

**Broad conversation export normalisation (Phase 1).** Claude Code JSONL and Claude.ai JSON are the formats that matter now. OpenAI ChatGPT export (`conversations.json`) is a stretch goal — the format is documented and the volume exists, but it's not blocking.

**Automatic entity extraction from free text.** KG entries are explicit writes. Implicit extraction is a later problem.

**Contradiction detection.** Ship it when it's actually wired up, not as a README claim.

---

## Data Model

### Context Taxonomy

Three metadata fields applied at ingest:

- **Domain** (`domain`): top-level context. A project name, a person's name, or a fixed category (`personal`, `technical`, `ops`). Analogous to MemPalace's wing.
- **Topic** (`topic`): subject within a domain. For projects: `auth`, `billing`, `deploy`, `infra`, `api`, `data`, `ci`, `security`. For conversations: `decisions`, `discoveries`, `preferences`, `problems`, `events`. Room + hall collapsed — one fewer filter dimension, no retrieval loss at this scale.
- **Source** (`source`): origin file path or session identifier. Used to link chunks back to the canonical store.

Additional chunk metadata: `timestamp` (ISO 8601), `importance` (float 0.0–1.0, default 0.5), `chunk_index`, `exchange_index`, `session_id`, `low_confidence_domain` (bool).

`importance` is a first-class tuning handle. Conversations that are only relevant in a narrow context should be filed with low importance (or 0.0 to suppress from L1 entirely). The hook may auto-score by recency + simple keyword density, but the Python API always accepts an explicit override.

### Chunking Strategy

**Conversations:** Exchange-pair chunking (user turn + assistant response = one chunk). If an exchange exceeds ~180 tokens (roughly 900 characters for English prose), sub-chunk with 100-character overlap, preserving `exchange_index` across all sub-chunks. Fall back to paragraph chunking if exchange markers are absent. Target chunk size: 150–200 tokens to stay inside 256-token embedding model limits. SHA-256 deduplication on raw text before embedding.

**Project files:** Path-first topic detection (file under `/auth/` → topic `auth`), then filename match, then keyword scoring against a fixed per-topic vocabulary. Recognised extensions:

`.py .js .ts .jsx .tsx .md .json .yaml .yml .toml .sql .sh .env .txt .rst .go .rs .java .rb .tf .hcl .proto .graphql .csv .xml`

Chunk at paragraph boundaries where possible; hard split at 200 tokens otherwise.

### SQLite Schema

Three responsibilities: knowledge graph, source tracking, and review queue.

```sql
-- Knowledge graph

CREATE TABLE entities (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,       -- person | project | system | concept
    properties  TEXT,                -- JSON blob
    created_at  TEXT NOT NULL
);

CREATE TABLE triples (
    id          INTEGER PRIMARY KEY,
    subject_id  INTEGER NOT NULL REFERENCES entities(id),
    predicate   TEXT NOT NULL,
    object_id   INTEGER REFERENCES entities(id),
    object_val  TEXT,                -- literal value when object is not an entity
    valid_from  TEXT NOT NULL,       -- ISO 8601
    valid_to    TEXT,                -- NULL = currently valid
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX idx_triples_subject ON triples(subject_id);
CREATE INDEX idx_triples_valid   ON triples(valid_from, valid_to);

-- Source tracking (incremental ingest + restic awareness)

CREATE TABLE sources (
    id                  INTEGER PRIMARY KEY,
    path                TEXT NOT NULL UNIQUE,  -- canonical path or URI
    format              TEXT NOT NULL,         -- claude_jsonl | claude_ai_json | openai_json | file
    content_hash        TEXT,                  -- hash of raw source at last ingest
    first_ingested_at   TEXT NOT NULL,
    last_ingested_at    TEXT NOT NULL,
    last_event_at       TEXT,                  -- timestamp of most recent event in this source
    chunk_count         INTEGER NOT NULL DEFAULT 0
);

-- Domain detection review queue

CREATE TABLE review_queue (
    id              INTEGER PRIMARY KEY,
    chunk_id        TEXT NOT NULL,             -- ChromaDB document ID
    guessed_domain  TEXT NOT NULL,
    guessed_topic   TEXT,
    confidence      REAL NOT NULL,
    raw_text        TEXT,                      -- first 500 chars for human review
    queued_at       TEXT NOT NULL,
    resolved_at     TEXT,
    resolved_domain TEXT
);

CREATE INDEX idx_review_unresolved ON review_queue(resolved_at)
    WHERE resolved_at IS NULL;
```

Point-in-time query pattern: `valid_from <= :as_of AND (valid_to IS NULL OR valid_to > :as_of)`.

The `sources` table serves two purposes: incremental ingest (skip re-parsing files whose `content_hash` hasn't changed) and providing restic with a consistent inventory — snapshot `~/.mnemon/` after any write quiesce and the table is the manifest.

### Canonical Store

Parsed, normalised conversation records live as JSON files under `~/.mnemon/canonical/` before chunking. Structure: one file per session, keyed by `session_id`. Each file is a list of exchange objects:

```json
{
  "session_id": "abc123",
  "source_format": "claude_jsonl",
  "source_path": "/path/to/original",
  "first_event_at": "2026-01-15T10:00:00Z",
  "last_event_at": "2026-01-15T11:23:00Z",
  "exchanges": [
    {
      "exchange_index": 0,
      "user": "...",
      "assistant": "...",
      "timestamp": "2026-01-15T10:01:00Z"
    }
  ]
}
```

Re-ingest is re-chunking the canonical form. The original export is only parsed once. Canonical files are what restic snapshots; ChromaDB embeddings are reproducible from them and do not need to be treated as the backup source of truth (though in practice you'd snapshot both).

---

## Vector Store Abstraction

ChromaDB is the Phase 1 implementation. Qdrant local mode is the likely Phase 4 upgrade at >100k chunks. Define the interface now so the swap is additive, not surgical.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float

class VectorStore(ABC):

    @abstractmethod
    def add(self, id: str, text: str, metadata: dict[str, Any]) -> None: ...

    @abstractmethod
    def search(
        self,
        query: str,
        n: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...

    @abstractmethod
    def get(self, id: str) -> SearchResult | None: ...

    @abstractmethod
    def list_metadata(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]: ...
```

`ChromaStore` and (later) `QdrantStore` implement this. The filter dict uses ChromaDB's `where` syntax internally; `QdrantStore` translates on the way in. Nothing above the store layer touches a ChromaDB type.

---

## Domain Resolver

Two implementations, selected by context:

**`StrictResolver`** — raises `DomainRequired` on missing or ambiguous domain. Used when the caller is a human-driven path (direct Python API calls, manual MCP tool invocation). Fail loudly; the caller knows the context and should tag it.

**`HeuristicResolver`** — makes a best guess using a keyword-to-domain map (comparable to MemPalace's ~60-entry room map), scores confidence, writes `low_confidence_domain=True` to chunk metadata, and appends to the `review_queue` table. Used by autonomous agents and hooks where stopping to ask is not an option.

The review queue is surfaced via `mnemon_review_list` (MCP tool, Phase 3) and a future `mnemon review` admin command. Resolving a queued item updates the chunk's metadata in ChromaDB and marks the row `resolved`.

Configuration selects the default resolver. Hooks use `HeuristicResolver`. The Python API defaults to `StrictResolver` and accepts an explicit override.

---

## Memory Stack

### L0 — Identity (~100 tokens)

Plain text at `~/.mnemon/identity.txt`. Static. Always prepended to wake-up output. Contains agent name, role, and invariant facts that should always be present without retrieval — the things whose absence would be incoherent.

### L1 — Critical Facts (~500–800 tokens)

Pre-computed. On every write, query the vector store for all chunks, sort descending by `importance`, take top 15, group by domain, truncate each to 200 characters, stop at 3200 total characters. Cache to `~/.mnemon/l1_cache.txt`. Batch the regeneration during bulk ingest (regenerate once at the end, not after each chunk).

This is the economic core of the system. The 250× cost difference between a pre-computed context blob and on-demand LLM summarisation is real and compounds daily.

### L2 — On-Demand Topic Recall (~200–500 tokens)

Vector store query with `domain` and/or `topic` filters. Return top 10 by relevance, each truncated to 300 characters. Agent-triggered when a topic becomes relevant mid-conversation.

### L3 — Deep Semantic Search

Full cosine similarity search, no filters, top 5 by distance. Fallback when L2 returns insufficient signal. Returned with scores so the caller can threshold.

---

## Ingest Pipeline

```
source file / conversation export
        │
        ▼
parse + normalise → canonical JSON (per session)
        │
        ├── write to ~/.mnemon/canonical/<session_id>.json
        ├── upsert sources table (hash, timestamps, count)
        │
        ▼
domain + topic detection (HeuristicResolver or StrictResolver)
        │
        ▼
chunk (exchange-pair or paragraph; sub-chunk if > ~180 tokens)
        │
        ▼
deduplicate (SHA-256 of raw text)
        │
        ▼
embed + store (VectorStore)
        │
        ▼
regenerate L1 cache (deferred during bulk ingest)
```

All stages deterministic and offline. No LLM calls at any stage.

---

## MCP Server

HTTP transport (HTTP+SSE), not stdio. Bind address configurable; default `127.0.0.1:7474`. Auth is a no-op middleware slot in Phase 1 — the interface accepts an `Authorization` header and ignores it. Adding bearer token validation in Phase 3 is additive, not structural.

Implemented with `fastmcp`.

### Tools

**Read:**
- `mnemon_status` — L0 + L1 + "know before speaking" protocol string
- `mnemon_wake_up` — L0 + L1 formatted for context injection
- `mnemon_search` — semantic search with optional `domain`/`topic` filters and `n`
- `mnemon_list_domains` — domains with chunk counts
- `mnemon_list_topics` — topics within a domain
- `mnemon_kg_query` — entity relationships, optional `as_of` for point-in-time
- `mnemon_kg_timeline` — chronological triple history for an entity

**Write:**
- `mnemon_add` — ingest a text chunk with domain, topic, importance, source
- `mnemon_kg_add` — add a triple
- `mnemon_kg_invalidate` — set `valid_to` on a triple

**Utility:**
- `mnemon_check_duplicate` — SHA-256 check before filing
- `mnemon_review_list` — surface low-confidence domain assignments (Phase 3)
- `mnemon_review_resolve` — confirm or correct a queued assignment (Phase 3)

Total: 13 tools (11 in Phase 1–2, +2 review tools in Phase 3).

---

## Claude Code Hooks

### Save Hook

Trigger: `Stop` event.

1. Serialise current conversation segment to a temp file using Claude Code's JSONL format.
2. Run ingest via Python subprocess (argument list, not `shell=True`).
3. L1 cache regenerates as part of ingest.
4. Append summary line to `~/.mnemon/saves.log`.

Importance weight for hook-triggered saves: recency-boosted (more recent exchanges score higher), capped at 0.8 to preserve headroom for manually-flagged high-importance items.

### Pre-Compact Hook

Trigger: `PreCompact` event.

Same as save hook, synchronous, with importance floor of 0.7 applied to the final exchange — if it's being compacted, something substantive happened.

### Hardening

- Python subprocess with argument list. No `shell=True`, no path interpolation.
- Validate session ID against `^[a-zA-Z0-9_-]{1,64}$` before use.
- Hard timeout: 30 seconds. Failure is logged and swallowed — never block Claude Code.
- Hook script is a thin shim; all logic lives in the Python package.

---

## Python API

```python
from mnemon import Memory

mem = Memory()  # config from ~/.mnemon/config.json

# Ingest — explicit domain (StrictResolver default)
mem.add(
    "Decided to use Clerk over Auth0: pricing and DX. Migration owned by Maya.",
    domain="hermes",
    topic="decisions",
    importance=0.9,
    source="session_abc123",
)

# Ingest — autonomous path (HeuristicResolver)
mem.add(text, resolver="heuristic")

# Search
results = mem.search("auth decisions", domain="hermes", n=5)

# Knowledge graph
mem.kg.add("hermes", "uses", "rabbitmq", valid_from="2026-01-01")
mem.kg.add("hermes", "deployed_on", "k3s", valid_from="2026-01-01")
mem.kg.query("hermes")
mem.kg.query("hermes", as_of="2026-02-01")
mem.kg.timeline("hermes")
mem.kg.invalidate("hermes", "uses", "rabbitmq", ended="2026-04-01")

# Wake-up context for context injection
print(mem.wake_up())           # L0 + L1
print(mem.wake_up(domain="hermes"))  # L0 + L1 filtered to domain

# Review queue
mem.review.list(limit=20)
mem.review.resolve(chunk_id="...", domain="hermes", topic="decisions")
```

---

## Storage Layout

```
~/.mnemon/
├── config.json              # bind address, domain_map, resolver default, etc.
├── identity.txt             # L0 — static identity block
├── l1_cache.txt             # pre-computed L1 context blob
├── knowledge_graph.db       # SQLite — entities, triples, sources, review_queue
├── saves.log                # hook save history
├── canonical/               # normalised conversation JSON, one file per session
│   └── <session_id>.json
└── chroma/                  # ChromaDB data directory
    └── mnemon_chunks/       # collection
```

Restic target: `~/.mnemon/`. ChromaDB writes are not atomic at the OS level; quiesce writes (or use the `mnemon_backup_prep` admin command, Phase 3) before snapshotting. The canonical store is the source of truth for re-ingest; the ChromaDB directory is a reproducible derivative.

---

## Dependencies

```toml
[project]
name = "mnemon"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.6.0,<0.8.0",   # pin upper bound; test before bumping
    "fastmcp>=0.1.0",
    "pyyaml>=6.0",
    "httpx>=0.27.0",             # HTTP client for inter-service calls if needed
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio",
    "ruff",
    "mypy",
]
```

No LangChain, no LlamaIndex, no Celery, no Redis. SQLite and ChromaDB are the full persistence stack. `uv` for everything.

---

## Non-Requirements (Explicit Exclusions)

- No AAAK or custom compression dialect
- No multi-user support or auth (Phase 1–2; auth hook in Phase 3)
- No cloud sync (restic handles backup; that's sufficient)
- No web UI
- No automatic entity extraction from free text
- No contradiction detection until it's actually implemented end-to-end
- No local LLM inference as a memory component
- No OpenAI ChatGPT export format (Phase 1–2; stretch goal Phase 4+)

---

## Open Questions

**Resolved:**

- ~~stdio vs HTTP MCP~~ → HTTP. Network-first, multi-machine.
- ~~Domain detection: strict vs heuristic~~ → Dual-mode via `DomainResolver`. Strict for human paths; heuristic with review queue for autonomous agents.
- ~~Session segmentation~~ → Not a chunking concern. Track session boundaries in metadata for L1 recency weighting; don't let it complicate chunk logic.
- ~~ChromaDB→Qdrant migration cost~~ → One day of interface work; data migration is the real cost. Mitigated by `VectorStore` abstraction.

**Open:**

1. **Importance auto-scoring in hooks.** Recency boost is described above. Should keyword density also contribute? Risk: heuristic complexity creep. Defer until there's real data from production use.

2. **L1 cache format.** Plain text blob is simplest. Structured JSON (domain → list of snippets) would allow the agent to navigate it more precisely. Plain text first; revisit when the L1 size becomes a problem.

3. **Canonical store format evolution.** The JSON schema above is v1. It will need versioning — add a `schema_version` field now, before there's anything to migrate.

4. **`mnemon_backup_prep` scope.** Needs to quiesce ChromaDB writes cleanly. ChromaDB doesn't expose an explicit flush API; the cleanest option may be a brief lock file that the ingest pipeline respects.

5. **Additional file extensions.** Current list covers the obvious cases. Worth a second pass once real project directories are being mined — there will be formats missing.

---

## Phased Delivery

**Phase 1 — Core (2–3 days)**  
`VectorStore` abstraction + `ChromaStore`, ingest pipeline (project files + Claude JSONL → canonical store), `StrictResolver`, `HeuristicResolver` + review queue write, Python API, L0/L1/L2/L3 memory stack, L1 cache generation. `sources` table. SHA-256 dedup.

**Phase 2 — KG + MCP (1–2 days)**  
SQLite knowledge graph (entities, triples, point-in-time queries, timeline), HTTP MCP server with 11 tools, `mnemon_status` with protocol injection, `mnemon_wake_up`.

**Phase 3 — Hooks + Hardening (1–2 days)**  
Claude Code save hook, pre-compact hook, hook hardening, `mnemon_review_list` + `mnemon_review_resolve` MCP tools, auth middleware slot, `mnemon_backup_prep`.

**Phase 4 — Integration + Stretch (ongoing)**  
Wire into Hermes. Tune importance scoring from real retrieval data. Qdrant swap if ChromaDB degrades at scale. OpenAI ChatGPT export format. `schema_version` migration tooling.

---

*v2 — incorporates design review: network MCP, VectorStore abstraction, dual-mode domain resolver, review queue, canonical store, SQLite sources table, revised chunking strategy, restic alignment, OpenAI format as stretch goal.*

---

Additional notes:

https://cookbook.chromadb.dev/strategies/backup/ Depending on your use case there are a few different ways to back up your ChromaDB data.
    API export - this approach is relatively simple, slow for large datasets and may result in a backup that is missing some updates, should your data change frequently.
    Disk snapshot - this approach is fast, but is highly dependent on the underlying storage. Should your cloud provider and underlying volume support snapshots, this is a good option.
    Filesystem backup - this approach is also fast, but requires stopping your Chroma container to avoid data corruption. This is a good option if you can afford to stop your Chroma container for a few minutes.

---

Good. That confirms the backup story is cleaner than it looked, because of a design decision already in the PRD.

**ChromaDB is not the source of truth — the canonical store is.** Embeddings are reproducible from the canonical JSON. That means the filesystem backup option (stop client, snapshot, restart) is the correct restic pattern, and if a snapshot is slightly dirty, the worst case is a rebuild from canonical rather than data loss. Worth stating explicitly in the doc.

In practice for Mnemon: `PersistentClient` (embedded mode, not server) means "stop container" translates to "close the client connection cleanly." ChromaDB uses SQLite with WAL mode internally, so a clean client close should flush the WAL before restic runs. The `mnemon_backup_prep` command can just be: close any open client handles, write a lock file, run restic, release lock. No quiesce API needed — clean close is sufficient.

The `chroma-datapipes` API export is worth knowing about but irrelevant here since canonical already gives us better fidelity than a ChromaDB JSONL export.

**Schema versioning in SQLite** — agreed, that's the right place. Standard pattern is a `schema_migrations` table:

```sql
CREATE TABLE schema_migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at  TEXT NOT NULL
);
```

One row per migration, applied sequentially. Version 1 is the initial schema. Anything that touches the SQLite schema goes through a migration. The canonical JSON files get a `schema_version` field too, but SQLite is authoritative — when there's a mismatch, the migration runner reconciles.

These are both small enough that they don't warrant a v3 of the PRD — I'd treat them as amendments to the open questions section when you move this into a proper spec. The backup approach resolves open question 4 (backup prep scope), and schema versioning closes the `schema_version` open question. Both resolved in favour of the simpler answer.

---

**Runtime**
- Python 3.11+
- ChromaDB `>=0.6.0,<0.8.0` — vector store (behind `VectorStore` ABC)
- FastMCP — HTTP MCP server
- PyYAML — config

**Storage**
- ChromaDB (embeddings + chunk metadata)
- SQLite (entities, triples, sources, review queue, schema migrations) — stdlib, no extra dep
- Local filesystem (`~/.mnemon/canonical/`, `~/.mnemon/` or perhaps `/mnemon` when dockerised)

**Dev**
- `uv` — everything
- `ruff` — linting
- `mypy` — type checking
- `pytest` + `pytest-asyncio` — tests
- `pip audit` and `bandit` or similar optional, but suggested.

**Notable absences by design**
- No LangChain / LlamaIndex
- No Redis
- No Celery
- No LLM calls at ingest time
- No cloud dependency at runtime

That's it. Two meaningful runtime dependencies outside stdlib if you squint at FastMCP as thin enough to almost not count. The SQLite KG, canonical store, and review queue are all stdlib `sqlite3`. The `VectorStore` abstraction means Qdrant can be added later without touching anything above the storage layer.
