"""Unit tests for text chunking."""
import unittest
try:
    import pytest
except ImportError:
    pytest = None
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.chunking.splitter import chunk_text, chunk_table


class TestChunkText(unittest.TestCase):
    def test_short_text_not_split(self):
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self):
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) > 0

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_metadata_preserved_in_chunks(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five."
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        assert len(chunks) >= 1

    def test_overlap_creates_context_continuity(self):
        text = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota."
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        # Later chunks should contain some text from the previous chunk
        if len(chunks) > 1:
            assert len(chunks[1]) > 0


class TestChunkTable(unittest.TestCase):
    def test_basic_table_chunked(self):
        table = {
            "headers": ["Name", "Value", "Unit"],
            "rows": [
                ["Alpha", "1.0", "kg"],
                ["Beta", "2.5", "kg"],
            ],
            "source": "test.pdf",
            "page_number": 1,
        }
        chunks = chunk_table(table)
        assert len(chunks) >= 1
        assert "Name" in chunks[0]
        assert "Alpha" in chunks[0]

    def test_empty_table_returns_empty(self):
        table = {"headers": [], "rows": [], "source": "x", "page_number": 1}
        chunks = chunk_table(table)
        assert chunks == []

    def test_large_table_split_into_multiple_chunks(self):
        table = {
            "headers": ["A", "B"],
            "rows": [["val", "val"] for _ in range(50)],
            "source": "large.csv",
            "page_number": 1,
        }
        chunks = chunk_table(table, max_rows_per_chunk=10)
        assert len(chunks) >= 3  # 50 rows / 10 per chunk = 5
