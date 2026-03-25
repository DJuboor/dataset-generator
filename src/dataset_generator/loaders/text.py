"""Plain text and markdown loader."""

from __future__ import annotations

import logging
from pathlib import Path

from dataset_generator.loaders.base import DocumentChunk, chunk_text

logger = logging.getLogger(__name__)


class TextLoader:
    """Load plain text and markdown files."""

    def load(
        self, path: Path, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[DocumentChunk]:
        """Read a text file and split into chunks.

        Tries UTF-8 first, falls back to latin-1 on decode errors.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s, falling back to latin-1", path)
            text = path.read_text(encoding="latin-1")

        return chunk_text(
            text, source=str(path), chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
