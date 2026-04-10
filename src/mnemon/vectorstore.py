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


def _default_embedding_function() -> Any:
    """Return ChromaDB's default embedding function explicitly."""
    return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()


def get_embedding_function(provider: str) -> Any:
    """Return a ChromaDB embedding function for the given provider name."""
    if provider == "default":
        return _default_embedding_function()
    if provider == "none":
        # Deterministic hash-based embeddings for testing.
        # Produces consistent 384-dim vectors from text content.
        import hashlib
        import struct

        class HashEmbeddingFunction:
            is_legacy = True

            def name(self) -> str:
                return "hash_test"

            def _embed(self, texts: list[str]) -> list[list[float]]:
                results = []
                for text in texts:
                    h = hashlib.sha256(text.encode()).digest()
                    floats = []
                    for i in range(384):
                        seed = hashlib.sha256(h + struct.pack(">I", i)).digest()[:4]
                        val = struct.unpack(">f", seed)[0]
                        floats.append(max(-1.0, min(1.0, val)))
                    results.append(floats)
                return results

            def __call__(self, input: list[str]) -> list[list[float]]:
                return self._embed(input)

            def embed_documents(self, input: list[str]) -> list[list[float]]:
                return self._embed(input)

            def embed_query(self, input: list[str]) -> list[list[float]]:
                return self._embed(input)

        return HashEmbeddingFunction()
    raise ValueError(f"Unknown embedding provider: {provider!r}. Use 'default' or 'none'.")


class ChromaStore(VectorStore):
    """Vector store backed by ChromaDB in embedded persistent mode.

    Parameters
    ----------
    persist_directory:
        Path to the ChromaDB data directory.
    collection_name:
        Name of the collection to use.
    embedding_function:
        ChromaDB ``EmbeddingFunction`` instance.  When *None* the default
        ``all-MiniLM-L6-v2`` function is used.  Pass an explicit function to
        control the provider (e.g. avoid ONNX/CoreML issues on macOS).
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "mnemon_chunks",
        embedding_function: Any = None,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
        )
        self._ef = embedding_function or _default_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
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
        for doc_id, doc_text, meta, dist in zip(
            ids[0], documents[0], metadatas[0], distances[0]
        ):
            out.append(
                SearchResult(
                    id=doc_id,
                    text=doc_text,
                    metadata=dict(meta),
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
            text=documents[0] or "",
            metadata=dict(metadatas[0]),
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
        raw = results.get("metadatas") or []
        # Also inject the document ID into each metadata dict so callers
        # can look up the full document via store.get(id).
        ids = results.get("ids") or []
        out: list[dict[str, Any]] = []
        for i, meta in enumerate(raw):
            d = dict(meta)
            if i < len(ids):
                d["id"] = ids[i]
            out.append(d)
        return out

    def count(self) -> int:
        return self._collection.count()
