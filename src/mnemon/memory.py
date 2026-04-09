"""Memory stack: L0 identity, L1 top-of-mind, L2/L3 search."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mnemon.config import MnemonConfig
from mnemon.vectorstore import SearchResult, VectorStore

PROTOCOL = (
    "Before answering questions about past decisions, projects,"
    " or preferences, search memory first using mnemon_search."
)


class MemoryStack:
    """Layered memory access for the Mnemon agent."""

    def __init__(self, store: VectorStore, config: MnemonConfig) -> None:
        self.store = store
        self.config = config

    # ── L0: identity ────────────────────────────────────────────

    def load_l0(self) -> str:
        """Read the identity file. Returns empty string if missing."""
        path = self.config.identity_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ── L1: top-of-mind cache ───────────────────────────────────

    def generate_l1(self) -> str:
        """Build the L1 top-of-mind summary and write it to the cache file."""
        metas = self.store.list_metadata(limit=1000)

        # Sort by importance descending, defaulting to 0.5.
        metas.sort(key=lambda m: m.get("importance", 0.5), reverse=True)
        top = metas[: self.config.l1_max_items]

        # Fetch full text and group by domain.
        grouped: dict[str, list[str]] = defaultdict(list)
        for meta in top:
            doc_id = meta.get("id", "")
            result = self.store.get(doc_id)
            if result is None:
                continue
            text = result.text[: self.config.l1_item_max_chars]
            domain = meta.get("domain", "general")
            grouped[domain].append(text)

        # Build formatted text, respecting max_chars.
        parts: list[str] = []
        total = 0
        for domain, items in grouped.items():
            header = f"## {domain}"
            if total + len(header) + 1 > self.config.l1_max_chars:
                break
            parts.append(header)
            total += len(header) + 1  # +1 for newline

            for item in items:
                line = f"- {item}"
                if total + len(line) + 1 > self.config.l1_max_chars:
                    break
                parts.append(line)
                total += len(line) + 1

        text = "\n".join(parts)
        self.config.l1_cache_path.write_text(text, encoding="utf-8")
        return text

    def load_l1(self) -> str:
        """Read the cached L1 text. Returns empty string if missing."""
        path = self.config.l1_cache_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ── wake_up ─────────────────────────────────────────────────

    def wake_up(self, domain: str | None = None) -> str:
        """Load L0 + L1, optionally filtering L1 to a single domain."""
        l0 = self.load_l0()
        l1 = self.load_l1()

        if domain and l1:
            l1 = self._filter_l1_by_domain(l1, domain)

        parts = [p for p in (l0, l1) if p]
        return "\n".join(parts)

    @staticmethod
    def _filter_l1_by_domain(l1_text: str, domain: str) -> str:
        """Return only the lines belonging to the given domain header."""
        lines = l1_text.splitlines()
        keep: list[str] = []
        in_domain = False
        for line in lines:
            if line.startswith("## "):
                in_domain = line == f"## {domain}"
            if in_domain:
                keep.append(line)
        return "\n".join(keep)

    # ── L2: filtered search ─────────────────────────────────────

    def search_l2(
        self,
        query: str,
        domain: str | None = None,
        topic: str | None = None,
        n: int = 10,
    ) -> list[SearchResult]:
        """Filtered vector search; results truncated to 300 chars."""
        where: dict[str, Any] | None = None
        filters: dict[str, Any] = {}
        if domain is not None:
            filters["domain"] = domain
        if topic is not None:
            filters["topic"] = topic
        if filters:
            where = filters

        results = self.store.search(query, n=n, where=where)
        for r in results:
            r.text = r.text[:300]
        return results

    # ── L3: deep search ─────────────────────────────────────────

    def search_l3(self, query: str, n: int = 5) -> list[SearchResult]:
        """Unfiltered vector search; full results with scores."""
        return self.store.search(query, n=n)

    # ── status ──────────────────────────────────────────────────

    def status(self) -> str:
        """Return L0 + L1 plus the memory-search protocol reminder."""
        l0 = self.load_l0()
        l1 = self.load_l1()
        parts = [p for p in (l0, l1) if p]
        parts.append(PROTOCOL)
        return "\n".join(parts)
