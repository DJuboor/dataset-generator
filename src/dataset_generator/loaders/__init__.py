"""Document loaders for --from-docs grounded generation."""

from __future__ import annotations

import logging
from pathlib import Path

from dataset_generator.loaders.base import DocumentChunk, DocumentLoader
from dataset_generator.loaders.text import TextLoader

__all__ = ["DocumentChunk", "DocumentLoader", "load_documents"]

logger = logging.getLogger(__name__)

LOADER_REGISTRY: dict[str, type[DocumentLoader]] = {
    ".txt": TextLoader,
    ".md": TextLoader,
}

# Register optional loaders (available when docs extra is installed)
try:
    from dataset_generator.loaders.pdf import PDFLoader

    LOADER_REGISTRY[".pdf"] = PDFLoader
except ImportError:
    pass

try:
    from dataset_generator.loaders.html import HTMLLoader

    LOADER_REGISTRY[".html"] = HTMLLoader
    LOADER_REGISTRY[".htm"] = HTMLLoader
except ImportError:
    pass


def load_documents(
    path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Load and chunk documents from a file or directory.

    Args:
        path: Path to a single file or a directory (searched recursively).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of DocumentChunk from all supported files found.
    """
    path = Path(path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(f for f in path.rglob("*") if f.is_file())
    else:
        logger.warning("Path does not exist: %s", path)
        return []

    chunks: list[DocumentChunk] = []
    for file in files:
        ext = file.suffix.lower()
        loader_cls = LOADER_REGISTRY.get(ext)
        if loader_cls is None:
            logger.warning("No loader for %s (skipping %s)", ext, file)
            continue

        try:
            loader = loader_cls()
            file_chunks = loader.load(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks.extend(file_chunks)
            logger.debug("Loaded %d chunks from %s", len(file_chunks), file)
        except Exception:
            logger.warning("Failed to load %s", file, exc_info=True)

    logger.info("Loaded %d total chunks from %d files", len(chunks), len(files))
    return chunks
