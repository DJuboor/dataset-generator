"""Tests for document loaders — chunking, text loader, and load_documents."""

from __future__ import annotations

import logging
from pathlib import Path

from dataset_generator.loaders.base import chunk_text
from dataset_generator.loaders.text import TextLoader

# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_splits_at_paragraph_boundaries(self):
        """Given text with paragraphs, chunks should split at double-newlines."""
        para_a = "A" * 400
        para_b = "B" * 400
        text = f"{para_a}\n\n{para_b}"

        chunks = chunk_text(text, source="test.txt", chunk_size=500, chunk_overlap=0)

        assert len(chunks) == 2
        assert para_a in chunks[0].text
        assert para_b in chunks[1].text

    def test_overlap_works(self):
        """Given overlap > 0, second chunk should start with tail of first chunk."""
        para_a = "A" * 400
        para_b = "B" * 400
        text = f"{para_a}\n\n{para_b}"

        chunks = chunk_text(text, source="test.txt", chunk_size=500, chunk_overlap=50)

        assert len(chunks) == 2
        # Second chunk should start with the last 50 chars of the first chunk
        assert chunks[1].text.startswith(para_a[-50:])

    def test_short_text_returns_single_chunk(self):
        """Given text shorter than chunk_size, should return a single chunk."""
        text = "Short text."

        chunks = chunk_text(text, source="file.txt", chunk_size=1000, chunk_overlap=100)

        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].source == "file.txt"

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", source="empty.txt", chunk_size=500, chunk_overlap=0)
        assert chunks == []

    def test_chunk_metadata(self):
        text = "Hello world"
        chunks = chunk_text(text, source="meta.txt", chunk_size=1000, chunk_overlap=0)
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_sentence_level_splitting(self):
        """When a single paragraph exceeds chunk_size, split at sentence boundaries."""
        # One giant paragraph with sentences
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_text(text, source="s.txt", chunk_size=40, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_word_level_splitting(self):
        """When no sentence boundaries exist, fall back to word splitting."""
        text = "word " * 100  # 500 chars, no sentence breaks
        chunks = chunk_text(text, source="w.txt", chunk_size=50, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_page_number_attached(self):
        text = "Some content here."
        chunks = chunk_text(text, source="f.txt", chunk_size=1000, chunk_overlap=0, page=3)
        assert chunks[0].page == 3


# ---------------------------------------------------------------------------
# TextLoader
# ---------------------------------------------------------------------------


class TestTextLoader:
    def test_loads_txt_file_and_chunks(self, tmp_path: Path):
        """Given a .txt file, TextLoader should read and chunk it."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        file_path = tmp_path / "sample.txt"
        file_path.write_text(content)

        loader = TextLoader()
        chunks = loader.load(file_path, chunk_size=5000, chunk_overlap=0)

        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "First paragraph" in combined
        assert "Third paragraph" in combined
        assert chunks[0].source == str(file_path)

    def test_latin1_fallback(self, tmp_path: Path):
        """Given a file with non-UTF-8 bytes, should fall back to latin-1."""
        file_path = tmp_path / "latin.txt"
        file_path.write_bytes(b"caf\xe9 cr\xe8me")

        loader = TextLoader()
        chunks = loader.load(file_path, chunk_size=5000, chunk_overlap=0)

        assert len(chunks) == 1
        assert "caf" in chunks[0].text


# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_loads_from_directory(self, tmp_path: Path):
        """Given a directory with .txt files, should load chunks from all of them."""
        from dataset_generator.loaders import load_documents

        (tmp_path / "a.txt").write_text("Content of file A.")
        (tmp_path / "b.txt").write_text("Content of file B.")

        chunks = load_documents(tmp_path, chunk_size=5000, chunk_overlap=0)

        assert len(chunks) >= 2
        texts = [c.text for c in chunks]
        assert any("file A" in t for t in texts)
        assert any("file B" in t for t in texts)

    def test_warns_on_unsupported_extension(self, tmp_path: Path, caplog):
        """Given a file with unsupported extension, should warn and skip it."""
        import logging

        from dataset_generator.loaders import load_documents

        (tmp_path / "data.xyz").write_text("unsupported format")

        with caplog.at_level(logging.WARNING):
            chunks = load_documents(tmp_path, chunk_size=5000, chunk_overlap=0)

        assert chunks == []
        assert any("No loader for" in msg for msg in caplog.messages)

    def test_loads_single_file(self, tmp_path: Path):
        """Given a path to a single file, should load it directly."""
        from dataset_generator.loaders import load_documents

        file_path = tmp_path / "single.txt"
        file_path.write_text("Single file content here.")

        chunks = load_documents(file_path, chunk_size=5000, chunk_overlap=0)

        assert len(chunks) == 1
        assert "Single file content" in chunks[0].text

    def test_nonexistent_path_returns_empty(self, tmp_path: Path, caplog):
        from dataset_generator.loaders import load_documents

        with caplog.at_level(logging.WARNING):
            chunks = load_documents(tmp_path / "does_not_exist", chunk_size=5000)

        assert chunks == []
        assert any("does not exist" in msg for msg in caplog.messages)

    def test_loader_exception_is_caught(self, tmp_path: Path, caplog):
        """If a loader raises, the file is skipped and others continue."""
        from unittest.mock import patch

        from dataset_generator.loaders import load_documents

        (tmp_path / "good.txt").write_text("Good content here.")
        (tmp_path / "bad.txt").write_text("Will fail.")

        original_load = TextLoader.load

        def _flaky_load(self, path, **kwargs):
            if "bad" in str(path):
                raise RuntimeError("simulated failure")
            return original_load(self, path, **kwargs)

        with (
            patch.object(TextLoader, "load", _flaky_load),
            caplog.at_level(logging.WARNING),
        ):
            chunks = load_documents(tmp_path, chunk_size=5000, chunk_overlap=0)

        assert len(chunks) >= 1
        assert any("Failed to load" in msg for msg in caplog.messages)
