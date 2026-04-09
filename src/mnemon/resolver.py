"""Domain resolution — classify text into domains and topics."""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnemon.db import Database


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DomainRequired(Exception):
    """Raised when a domain cannot be inferred and must be supplied explicitly."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ResolverResult:
    """Outcome of a domain-resolution attempt."""

    domain: str
    topic: str | None
    confidence: float
    low_confidence: bool


# ---------------------------------------------------------------------------
# Default keyword maps
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "engineering": [
        "code", "refactor", "function", "class", "module", "variable",
        "compile", "runtime", "debug", "stack trace", "pull request",
        "merge", "branch", "commit", "repository", "linter",
    ],
    "infrastructure": [
        "server", "docker", "container", "k8s", "kubernetes", "terraform",
        "cloud", "aws", "gcp", "azure", "vm", "instance", "cluster",
        "load balancer", "cdn", "dns",
    ],
    "data": [
        "database", "migration", "schema", "query", "sql", "etl",
        "pipeline", "warehouse", "analytics", "metric", "dashboard",
        "postgres", "redis", "mongo",
    ],
    "security": [
        "vulnerability", "cve", "audit", "encryption", "ssl", "tls",
        "firewall", "penetration", "rbac", "permission", "secret",
    ],
    "product": [
        "feature", "requirement", "user story", "roadmap", "milestone",
        "backlog", "sprint", "stakeholder", "customer", "feedback",
    ],
    "operations": [
        "incident", "alert", "monitoring", "sla", "uptime", "runbook",
        "on-call", "escalation", "postmortem", "outage",
    ],
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "auth": ["auth", "login", "session", "token", "oauth", "jwt", "password", "credential"],
    "billing": ["billing", "payment", "invoice", "subscription", "stripe", "charge", "price"],
    "deploy": ["deploy", "release", "rollout", "ci/cd", "pipeline", "staging", "production"],
    "infra": ["infra", "server", "docker", "k8s", "kubernetes", "terraform", "cloud", "aws", "gcp"],
    "api": ["api", "endpoint", "route", "rest", "graphql", "request", "response"],
    "data": ["data", "database", "migration", "schema", "query", "sql", "model"],
    "ci": ["ci", "test", "build", "lint", "github actions", "workflow"],
    "security": ["security", "vulnerability", "cve", "audit", "encryption", "ssl", "tls"],
    "decisions": ["decided", "decision", "chose", "choice", "approach", "strategy"],
    "discoveries": ["found", "discovered", "learned", "realized", "noticed", "turns out"],
    "preferences": ["prefer", "preference", "like", "dislike", "convention", "style"],
    "problems": ["bug", "error", "issue", "problem", "broken", "failing", "crash"],
    "events": ["meeting", "standup", "review", "demo", "launch", "deadline", "incident"],
}

# Path components that map directly to topics.
_PATH_TOPIC_PATTERNS: dict[str, str] = {
    "auth": "auth",
    "billing": "billing",
    "deploy": "deploy",
    "infra": "infra",
    "api": "api",
    "data": "data",
    "ci": "ci",
    "security": "security",
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DomainResolver(ABC):
    """Strategy interface for mapping raw text to a domain + topic."""

    @abstractmethod
    def resolve(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ResolverResult:
        """Return a *ResolverResult* for *text*."""


# ---------------------------------------------------------------------------
# Strict resolver — requires explicit domain
# ---------------------------------------------------------------------------


class StrictResolver(DomainResolver):
    """Only accepts an explicit ``domain`` key in *metadata*."""

    def resolve(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ResolverResult:
        metadata = metadata or {}
        domain = metadata.get("domain")
        if domain is None:
            raise DomainRequired("StrictResolver requires 'domain' in metadata")
        return ResolverResult(
            domain=str(domain),
            topic=metadata.get("topic"),
            confidence=1.0,
            low_confidence=False,
        )


# ---------------------------------------------------------------------------
# Heuristic resolver — keyword + path scoring
# ---------------------------------------------------------------------------


class HeuristicResolver(DomainResolver):
    """Score-based resolver using keyword matching and file-path heuristics."""

    def __init__(
        self,
        domain_map: dict[str, list[str]] | None = None,
        db: Database | None = None,
    ) -> None:
        self.domain_map = domain_map or DOMAIN_KEYWORDS
        self.db = db

    # -- public API --------------------------------------------------------

    def resolve(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ResolverResult:
        metadata = metadata or {}

        # (a) Explicit domain in metadata — trust it fully.
        if "domain" in metadata:
            topic = metadata.get("topic") or self._detect_topic(text, metadata)
            return ResolverResult(
                domain=str(metadata["domain"]),
                topic=topic,
                confidence=1.0,
                low_confidence=False,
            )

        # (b) Path-based topic hint (may also inform domain scoring).
        path_topic = self._topic_from_path(metadata.get("source"))

        # (c) Keyword scoring for domain.
        scores = self._score_domains(text)
        if not scores:
            raise DomainRequired("HeuristicResolver could not determine a domain")

        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_domain, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        # (d) Confidence = separation ratio.
        confidence = (top_score - second_score) / max(top_score, 1)

        # (e) Low-confidence flag.
        low_confidence = confidence < 0.6

        # Topic detection — path first, then keyword density.
        topic = path_topic or self._detect_topic(text, metadata)

        result = ResolverResult(
            domain=top_domain,
            topic=topic,
            confidence=round(confidence, 4),
            low_confidence=low_confidence,
        )

        # Write to review queue when confidence is low.
        if low_confidence and self.db is not None:
            self._enqueue_review(result, text)

        return result

    # -- internal helpers --------------------------------------------------

    def _score_domains(self, text: str) -> dict[str, int]:
        lower = text.lower()
        scores: dict[str, int] = {}
        for domain, keywords in self.domain_map.items():
            count = sum(1 for kw in keywords if kw in lower)
            if count > 0:
                scores[domain] = count
        return scores

    @staticmethod
    def _topic_from_path(source: Any) -> str | None:
        if not source or not isinstance(source, str):
            return None
        parts = re.split(r"[\\/]", source.lower())
        for part in parts:
            if part in _PATH_TOPIC_PATTERNS:
                return _PATH_TOPIC_PATTERNS[part]
        return None

    @staticmethod
    def _detect_topic(text: str, metadata: dict[str, Any]) -> str | None:
        # Try path first.
        path_topic = HeuristicResolver._topic_from_path(metadata.get("source"))
        if path_topic:
            return path_topic

        # Fall back to keyword density across TOPIC_KEYWORDS.
        lower = text.lower()
        best_topic: str | None = None
        best_count = 0
        for topic, keywords in TOPIC_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in lower)
            if count > best_count:
                best_count = count
                best_topic = topic
        return best_topic

    def _enqueue_review(self, result: ResolverResult, text: str) -> None:
        assert self.db is not None
        self.db.execute(
            """
            INSERT INTO review_queue (chunk_id, guessed_domain, guessed_topic,
                                      confidence, raw_text, queued_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                result.domain,
                result.topic,
                result.confidence,
                text,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.db.commit()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_resolver(name: str, **kwargs: Any) -> DomainResolver:
    """Return a *DomainResolver* instance by short name.

    Supported names: ``"strict"``, ``"heuristic"``.
    """
    resolvers: dict[str, type[DomainResolver]] = {
        "strict": StrictResolver,
        "heuristic": HeuristicResolver,
    }
    cls = resolvers.get(name)
    if cls is None:
        raise ValueError(f"Unknown resolver: {name!r} (available: {sorted(resolvers)})")
    return cls(**kwargs)
