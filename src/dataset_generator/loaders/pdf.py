"""PDF loader using PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path

from dataset_generator.loaders.base import DocumentChunk, chunk_text

logger = logging.getLogger(__name__)

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None  # type: ignore[assignment]


class PDFLoader:
    """Load PDF files page by page using PyMuPDF."""

    def load(
        self, path: Path, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[DocumentChunk]:
        """Extract text from each PDF page and chunk it.

        Requires the ``docs`` extra: ``uv add dataset-generator[docs]``.
        """
        if fitz is None:
            raise ImportError("pymupdf is required for PDF loading. Install with: uv add pymupdf")

        chunks: list[DocumentChunk] = []
        source = str(path)

        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()  # type: ignore[union-attr]
                if not text.strip():
                    continue
                page_chunks = chunk_text(
                    text,
                    source=source,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    page=page_num,
                )
                chunks.extend(page_chunks)

        logger.debug("Loaded %d chunks from %d-page PDF: %s", len(chunks), page_num, path)
        return chunks
