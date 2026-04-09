"""Tests for mnemon.resolver — domain and topic classification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnemon.db import Database
from mnemon.resolver import (
    DomainRequired,
    HeuristicResolver,
    ResolverResult,
    StrictResolver,
    get_resolver,
)


# ---------------------------------------------------------------------------
# StrictResolver
# ---------------------------------------------------------------------------


class TestStrictResolver:
    def test_raises_when_no_domain_in_metadata(self) -> None:
        resolver = StrictResolver()
        with pytest.raises(DomainRequired):
            resolver.resolve("some text")

    def test_raises_when_metadata_empty(self) -> None:
        resolver = StrictResolver()
        with pytest.raises(DomainRequired):
            resolver.resolve("some text", metadata={})

    def test_passes_through_explicit_domain(self) -> None:
        resolver = StrictResolver()
        result = resolver.resolve("anything", metadata={"domain": "billing"})
        assert result.domain == "billing"
        assert result.confidence == 1.0
        assert result.low_confidence is False

    def test_passes_through_domain_and_topic(self) -> None:
        resolver = StrictResolver()
        result = resolver.resolve(
            "anything",
            metadata={"domain": "engineering", "topic": "auth"},
        )
        assert result.domain == "engineering"
        assert result.topic == "auth"


# ---------------------------------------------------------------------------
# HeuristicResolver — keyword detection
# ---------------------------------------------------------------------------


class TestHeuristicResolverKeywords:
    def test_detects_engineering_domain(self) -> None:
        resolver = HeuristicResolver()
        result = resolver.resolve(
            "Refactored the function to avoid a stack trace in the module"
        )
        assert result.domain == "engineering"
        assert result.confidence > 0

    def test_detects_infrastructure_domain(self) -> None:
        resolver = HeuristicResolver()
        result = resolver.resolve(
            "Deployed a new docker container on the kubernetes cluster in aws"
        )
        assert result.domain == "infrastructure"

    def test_explicit_domain_in_metadata_overrides(self) -> None:
        resolver = HeuristicResolver()
        result = resolver.resolve(
            "Docker container on kubernetes",
            metadata={"domain": "custom"},
        )
        assert result.domain == "custom"
        assert result.confidence == 1.0

    def test_raises_when_no_keywords_match(self) -> None:
        resolver = HeuristicResolver()
        with pytest.raises(DomainRequired):
            resolver.resolve("xyzzy plugh abracadabra")


# ---------------------------------------------------------------------------
# HeuristicResolver — path-based topic detection
# ---------------------------------------------------------------------------


class TestHeuristicResolverPathTopic:
    def test_detects_auth_topic_from_path(self) -> None:
        resolver = HeuristicResolver()
        result = resolver.resolve(
            "Refactored the function and class",
            metadata={"source": "/repo/src/auth/middleware.py"},
        )
        assert result.topic == "auth"

    def test_detects_billing_topic_from_path(self) -> None:
        resolver = HeuristicResolver()
        result = resolver.resolve(
            "Updated the module code",
            metadata={"source": "/repo/billing/service.py"},
        )
        assert result.topic == "billing"

    def test_falls_back_to_keyword_topic(self) -> None:
        resolver = HeuristicResolver()
        result = resolver.resolve(
            "Fixed the login session token code in the module"
        )
        assert result.topic == "auth"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    def test_high_confidence_with_clear_winner(self) -> None:
        resolver = HeuristicResolver()
        # Lots of engineering keywords, nothing else.
        result = resolver.resolve(
            "Refactored the function class module variable compile runtime debug"
        )
        assert result.confidence >= 0.6
        assert result.low_confidence is False

    def test_low_confidence_when_ambiguous(self) -> None:
        resolver = HeuristicResolver()
        # Mix keywords from two domains equally.
        result = resolver.resolve(
            "server docker container code function class"
        )
        assert result.low_confidence is True
        assert result.confidence < 0.6


# ---------------------------------------------------------------------------
# Review queue write on low confidence
# ---------------------------------------------------------------------------


class TestReviewQueueWrite:
    def test_writes_to_review_queue_on_low_confidence(self) -> None:
        mock_db = MagicMock(spec=Database)
        resolver = HeuristicResolver(db=mock_db)

        # Ambiguous text to trigger low confidence.
        resolver.resolve("server docker code function class container")

        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "review_queue" in sql
        mock_db.commit.assert_called_once()

    def test_no_write_when_high_confidence(self) -> None:
        mock_db = MagicMock(spec=Database)
        resolver = HeuristicResolver(db=mock_db)

        resolver.resolve(
            "Refactored the function class module variable compile runtime debug"
        )

        mock_db.execute.assert_not_called()

    def test_no_write_when_db_not_provided(self) -> None:
        resolver = HeuristicResolver()  # no db
        # Just make sure it doesn't raise.
        result = resolver.resolve("server docker code function class container")
        assert result.low_confidence is True

    def test_review_queue_with_real_db(self, tmp_path: Path) -> None:
        db = Database(path=tmp_path / "test.db")
        resolver = HeuristicResolver(db=db)

        resolver.resolve("server docker code function class container")

        row = db.fetchone(
            "SELECT * FROM review_queue WHERE resolved_at IS NULL"
        )
        assert row is not None
        assert row["confidence"] < 0.6
        db.close()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestGetResolver:
    def test_returns_strict_resolver(self) -> None:
        resolver = get_resolver("strict")
        assert isinstance(resolver, StrictResolver)

    def test_returns_heuristic_resolver(self) -> None:
        resolver = get_resolver("heuristic")
        assert isinstance(resolver, HeuristicResolver)

    def test_raises_on_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown resolver"):
            get_resolver("magic")

    def test_passes_kwargs_to_heuristic(self) -> None:
        custom_map = {"custom_domain": ["keyword1", "keyword2"]}
        resolver = get_resolver("heuristic", domain_map=custom_map)
        assert isinstance(resolver, HeuristicResolver)
        assert resolver.domain_map == custom_map
