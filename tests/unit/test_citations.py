"""Unit tests for citation validation."""
import unittest
try:
    import pytest
except ImportError:
    pytest = None
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.generation.citations import validate_citations, format_context, format_source_list


def make_source(num, source="doc.pdf", page=1):
    return {"citation_num": num, "source": source, "page_number": page, "chunk_id": f"c{num}", "score": 0.9}


class TestValidateCitations(unittest.TestCase):
    def test_valid_citations_unchanged(self):
        answer = "This is true [1] and also [2]."
        sources = [make_source(1), make_source(2)]
        validated, warnings = validate_citations(answer, sources)
        assert "[1:INVALID]" not in validated
        assert "[2:INVALID]" not in validated
        assert len(warnings) == 0

    def test_invalid_citation_flagged(self):
        answer = "This is wrong [5]."
        sources = [make_source(1)]
        validated, warnings = validate_citations(answer, sources)
        assert "[5:INVALID]" in validated
        assert len(warnings) == 1

    def test_no_citations_no_warnings(self):
        answer = "No citations here."
        sources = [make_source(1)]
        _, warnings = validate_citations(answer, sources)
        assert len(warnings) == 0


class TestFormatContext(unittest.TestCase):
    def test_formats_numbered_sources(self):
        results = [
            {"chunk_id": "c1", "text": "Some text", "metadata": {"source": "doc.pdf", "page_number": 1}, "score": 0.9},
        ]
        context, sources = format_context(results)
        assert "[1]" in context
        assert "doc.pdf" in context
        assert len(sources) == 1
        assert sources[0]["citation_num"] == 1
