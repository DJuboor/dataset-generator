"""Base types for document loaders."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a source document."""

    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(Protocol):
    """Protocol for document loaders."""

    def load(
        self, path: Path, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[DocumentChunk]: ...


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page: int | None = None,
) -> list[DocumentChunk]:
    """Split text into overlapping chunks at natural boundaries.

    Tries paragraph boundaries first, then sentences, then words.

    Args:
        text: Full text to chunk.
        source: Source file path for attribution.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        page: Optional page number to attach to all chunks.
    """
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\n+", text.strip())
    segments = _merge_segments(paragraphs, chunk_size)

    # If any segment is still too large, split further at sentence boundaries
    refined: list[str] = []
    for seg in segments:
        if len(seg) <= chunk_size:
            refined.append(seg)
        else:
            sentences = re.split(r"(?<=\. )", seg)
            refined.extend(_merge_segments(sentences, chunk_size))

    # Final pass: split anything still too large at word boundaries
    final: list[str] = []
    for seg in refined:
        if len(seg) <= chunk_size:
            final.append(seg)
        else:
            words = seg.split(" ")
            final.extend(_merge_segments(words, chunk_size, join_sep=" "))

    return _apply_overlap(final, source, chunk_size, chunk_overlap, page)


def _merge_segments(parts: list[str], max_size: int, join_sep: str = "\n\n") -> list[str]:
    """Greedily merge small parts into segments up to max_size."""
    segments: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{join_sep}{part}" if current else part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                segments.append(current)
            current = part
    if current:
        segments.append(current)
    return segments


def _apply_overlap(
    segments: list[str],
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    page: int | None,
) -> list[DocumentChunk]:
    """Build DocumentChunks with overlap from ordered segments."""
    if not segments:
        return []

    chunks: list[DocumentChunk] = []
    for i, seg in enumerate(segments):
        if i > 0 and chunk_overlap > 0:
            prev = segments[i - 1]
            overlap_text = prev[-chunk_overlap:]
            seg = overlap_text + seg

        chunks.append(
            DocumentChunk(
                text=seg,
                source=source,
                page=page,
                metadata={"chunk_index": i, "total_chunks": len(segments)},
            )
        )
    return chunks
