# Mnemon

Persistent memory layer for agentic workflows. No API keys, no cloud dependencies, no LLM calls at runtime.

Exposes a **Python API** and an **HTTP MCP server** for any MCP-compatible agent. Primary consumers: Claude Code (via hooks + MCP), Hermes meta-agent.

## Architecture

**4-layer memory stack** with progressive context loading:

| Layer | Size | Loaded | Description |
|-------|------|--------|-------------|
| L0 | ~100 tokens | Always | Identity (`~/.mnemon/identity.txt`) |
| L1 | ~500-800 tokens | Always | Pre-computed top-N important memories |
| L2 | ~200-500 tokens | On demand | Topic-filtered vector search |
| L3 | Unbounded | On demand | Full semantic search |

Wake-up cost under 900 tokens.

**Storage stack:**

- **ChromaDB** -- vector embeddings + chunk metadata (behind a `VectorStore` abstraction for future Qdrant swap)
- **SQLite** -- temporal knowledge graph (entities + triples with `valid_from`/`valid_to`), source tracking, review queue
- **Filesystem** -- canonical conversation JSON (`~/.mnemon/canonical/`), config, caches

**Ingest pipeline:** parse -> canonical store -> domain/topic resolution -> exchange-pair chunking -> SHA-256 dedup -> embed -> L1 cache regeneration. All stages deterministic and offline.

## Install

```bash
uv sync
```

Requires Python 3.11+.

## Python API

```python
from mnemon import Memory

mem = Memory()  # config from ~/.mnemon/config.json

# Store a memory
mem.add(
    "Decided to use Clerk over Auth0: pricing and DX.",
    domain="hermes",
    topic="decisions",
    importance=0.9,
)

# Search
results = mem.search("auth decisions", domain="hermes", n=5)

# Knowledge graph
mem.kg.add("hermes", "uses", "rabbitmq", valid_from="2026-01-01")
mem.kg.query("hermes")
mem.kg.query("hermes", as_of="2026-02-01")  # point-in-time
mem.kg.timeline("hermes")
mem.kg.invalidate("hermes", "uses", "rabbitmq", ended="2026-04-01")

# Wake-up context for injection
print(mem.wake_up())           # L0 + L1
print(mem.wake_up(domain="hermes"))

# Ingest conversation exports
mem.ingest("session.jsonl")             # Claude Code JSONL
mem.ingest("conversation.json")         # Claude.ai JSON

# Domain detection review queue
mem.review.list(limit=20)
mem.review.resolve(chunk_id="...", domain="hermes", topic="decisions")
```

## MCP Server

HTTP transport on `127.0.0.1:7474` (configurable).

```bash
uv run python -m mnemon.server
```

### Tools

| Tool | Type | Description |
|------|------|-------------|
| `mnemon_status` | Read | L0 + L1 + search-first protocol |
| `mnemon_wake_up` | Read | L0 + L1, optional domain filter |
| `mnemon_search` | Read | Semantic search with domain/topic filters |
| `mnemon_list_domains` | Read | Domains with chunk counts |
| `mnemon_list_topics` | Read | Topics within a domain |
| `mnemon_kg_query` | Read | Entity relationships, optional point-in-time |
| `mnemon_kg_timeline` | Read | Chronological triple history |
| `mnemon_add` | Write | Ingest a text chunk |
| `mnemon_kg_add` | Write | Add a knowledge graph triple |
| `mnemon_kg_invalidate` | Write | End-date a triple |
| `mnemon_check_duplicate` | Utility | SHA-256 dedup check |

## Claude Code Hooks

Save hook (Stop event) and pre-compact hook (PreCompact event) for automatic memory capture:

```bash
python -m mnemon.hooks stop /path/to/conversation.jsonl session_id
python -m mnemon.hooks pre_compact /path/to/conversation.jsonl session_id
```

Hardened: argument list (no `shell=True`), validated session IDs, 30-second timeout, failures logged and swallowed.

## Domain Resolution

Two modes:

- **StrictResolver** -- requires explicit domain. Used for human-driven paths. Raises `DomainRequired` on missing domain.
- **HeuristicResolver** -- keyword + path-based scoring with confidence thresholds. Low-confidence assignments go to a review queue. Used by hooks and autonomous agents.

## Storage Layout

```
~/.mnemon/
  config.json           # bind address, domain_map, resolver default
  identity.txt          # L0 static identity block
  l1_cache.txt          # pre-computed L1 context
  knowledge_graph.db    # SQLite (entities, triples, sources, review_queue)
  saves.log             # hook save history
  canonical/            # normalised conversation JSON per session
  chroma/               # ChromaDB data directory
```

## Configuration

`~/.mnemon/config.json`:

```json
{
  "bind_host": "127.0.0.1",
  "bind_port": 7474,
  "default_resolver": "strict",
  "chroma_collection": "mnemon_chunks",
  "domain_map": {}
}
```

## Tests

```bash
uv run pytest
```

## Dependencies

- `chromadb` -- vector store
- `fastmcp` -- MCP server
- `pyyaml` -- config
- `httpx` -- HTTP client

No LangChain, no LlamaIndex, no Redis, no Celery, no cloud runtime dependencies.
