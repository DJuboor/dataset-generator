"""HTML loader using BeautifulSoup."""

from __future__ import annotations

import logging
from pathlib import Path

from dataset_generator.loaders.base import DocumentChunk, chunk_text

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment, misc]


class HTMLLoader:
    """Load HTML files, stripping tags to extract text content."""

    def load(
        self, path: Path, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[DocumentChunk]:
        """Parse HTML, extract text, and chunk it.

        Requires the ``docs`` extra: ``uv add dataset-generator[docs]``.
        """
        if BeautifulSoup is None:
            raise ImportError(
                "beautifulsoup4 is required for HTML loading. Install with: uv add beautifulsoup4"
            )

        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s, falling back to latin-1", path)
            raw = path.read_text(encoding="latin-1")

        soup = BeautifulSoup(raw, "html.parser")

        # Remove script and style elements before extracting text
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n\n")

        return chunk_text(
            text, source=str(path), chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
