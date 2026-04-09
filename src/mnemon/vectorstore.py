"""Vector store abstraction for Mnemon."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


@dataclass
class SearchResult:
    """A single result from a vector store search."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, id: str, text: str, metadata: dict[str, Any]) -> None:
        """Add a document to the store."""

    @abstractmethod
    def search(
        self,
        query: str,
        n: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a document by ID."""

    @abstractmethod
    def get(self, id: str) -> SearchResult | None:
        """Get a document by ID, or None if not found."""

    @abstractmethod
    def list_metadata(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """List metadata dicts, optionally filtered."""

    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self.list_metadata())


class ChromaStore(VectorStore):
    """Vector store backed by ChromaDB in embedded persistent mode."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "mnemon_chunks",
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    def add(self, id: str, text: str, metadata: dict[str, Any]) -> None:
        self._collection.add(ids=[id], documents=[text], metadatas=[metadata])

    def search(
        self,
        query: str,
        n: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        ids = results.get("ids") or [[]]
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        out: list[SearchResult] = []
        for doc_id, text, meta, dist in zip(
            ids[0], documents[0], metadatas[0], distances[0]
        ):
            out.append(
                SearchResult(
                    id=doc_id,
                    text=text,
                    metadata=meta,
                    score=1.0 - dist,
                )
            )
        return out

    def delete(self, id: str) -> None:
        self._collection.delete(ids=[id])

    def get(self, id: str) -> SearchResult | None:
        results = self._collection.get(ids=[id])
        ids = results.get("ids") or []
        if not ids:
            return None
        documents = results.get("documents") or [""]
        metadatas = results.get("metadatas") or [{}]
        return SearchResult(
            id=ids[0],
            text=documents[0],
            metadata=metadatas[0],
            score=0.0,
        )

    def list_metadata(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"limit": limit}
        if where:
            kwargs["where"] = where
        results = self._collection.get(**kwargs)
        return list(results.get("metadatas") or [])

    def count(self) -> int:
        return self._collection.count()
