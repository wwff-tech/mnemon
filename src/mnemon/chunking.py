"""Text chunking utilities for Mnemon memory ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A chunk of text produced by splitting exchanges or documents."""

    text: str
    chunk_index: int
    exchange_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~5 chars per token heuristic.

    PRD guideline: ~180 tokens ≈ 900 chars → 5 chars/token.
    """
    length = len(text)
    return length // 5 + (1 if length % 5 else 0)


def compute_chunk_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()


def sub_chunk(text: str, max_tokens: int = 180, overlap_chars: int = 100) -> list[str]:
    """Split *text* into overlapping segments of roughly *max_tokens* tokens.

    Each segment is at most ``max_tokens * 5`` characters long and overlaps
    with the next segment by *overlap_chars* characters.
    """
    max_chars = max_tokens * 5
    if max_chars <= 0:
        return [text] if text else []
    if len(text) <= max_chars:
        return [text] if text else []

    segments: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        segments.append(text[start:end])
        # Advance by (max_chars - overlap_chars) so the next segment overlaps.
        step = max_chars - overlap_chars
        if step <= 0:
            step = 1  # safety: always advance
        start += step
    return segments


def chunk_exchanges(
    exchanges: list[dict[str, Any]],
    max_tokens: int = 180,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """Chunk a list of conversation exchanges.

    Each exchange dict is expected to contain:
    - ``exchange_index`` (int)
    - ``user`` (str)
    - ``assistant`` (str)
    - ``timestamp`` (str, optional)

    Returns a flat list of :class:`Chunk` objects with sequential
    ``chunk_index`` values starting from 0.
    """
    chunks: list[Chunk] = []
    global_index = 0

    for exchange in exchanges:
        ex_index: int = exchange["exchange_index"]
        text = exchange["user"] + "\n\n" + exchange["assistant"]
        timestamp = exchange.get("timestamp")

        if estimate_tokens(text) <= max_tokens:
            meta: dict[str, Any] = {}
            if timestamp is not None:
                meta["timestamp"] = timestamp
            chunks.append(
                Chunk(
                    text=text,
                    chunk_index=global_index,
                    exchange_index=ex_index,
                    metadata=meta,
                    content_hash=compute_chunk_hash(text),
                )
            )
            global_index += 1
        else:
            segments = sub_chunk(text, max_tokens=max_tokens, overlap_chars=overlap_chars)
            for segment in segments:
                meta = {}
                if timestamp is not None:
                    meta["timestamp"] = timestamp
                chunks.append(
                    Chunk(
                        text=segment,
                        chunk_index=global_index,
                        exchange_index=ex_index,
                        metadata=meta,
                        content_hash=compute_chunk_hash(segment),
                    )
                )
                global_index += 1

    return chunks


def chunk_paragraphs(text: str, max_tokens: int = 200) -> list[Chunk]:
    """Chunk free-form text by paragraph boundaries.

    - Splits on double newlines.
    - Merges consecutive short paragraphs until approaching *max_tokens*.
    - Hard-splits any single paragraph that exceeds *max_tokens*.
    - Chunks have no ``exchange_index``.
    """
    if not text:
        return []

    paragraphs = text.split("\n\n")
    max_chars = max_tokens * 5

    chunks: list[Chunk] = []
    buffer = ""

    def _flush(buf: str) -> None:
        if not buf:
            return
        # Hard-split if the buffer itself exceeds max_chars.
        if len(buf) <= max_chars:
            chunks.append(
                Chunk(
                    text=buf,
                    chunk_index=len(chunks),
                    content_hash=compute_chunk_hash(buf),
                )
            )
        else:
            # Hard-split without overlap (paragraph chunking doesn't use overlap).
            start = 0
            while start < len(buf):
                segment = buf[start : start + max_chars]
                chunks.append(
                    Chunk(
                        text=segment,
                        chunk_index=len(chunks),
                        content_hash=compute_chunk_hash(segment),
                    )
                )
                start += max_chars

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if buffer:
            candidate = buffer + "\n\n" + para
        else:
            candidate = para

        if estimate_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            # Flush existing buffer first, then start fresh with this paragraph.
            _flush(buffer)
            buffer = para

    _flush(buffer)
    return chunks
