"""Unit tests for Reciprocal Rank Fusion."""
import unittest
try:
    import pytest
except ImportError:
    pytest = None
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.retrieval.fusion import reciprocal_rank_fusion


def make_result(chunk_id: str, text: str = "text", score: float = 0.9):
    return {"chunk_id": chunk_id, "text": text, "score": score, "metadata": {}}


class TestRRF(unittest.TestCase):
    def test_single_list_preserved(self):
        results = [make_result("a"), make_result("b"), make_result("c")]
        fused = reciprocal_rank_fusion([results], top_k=3)
        assert len(fused) == 3
        assert all("rrf_score" in r for r in fused)

    def test_deduplication(self):
        list1 = [make_result("a"), make_result("b")]
        list2 = [make_result("a"), make_result("c")]
        fused = reciprocal_rank_fusion([list1, list2], top_k=10)
        ids = [r["chunk_id"] for r in fused]
        assert len(ids) == len(set(ids)), "Duplicates found"

    def test_shared_top_result_ranks_higher(self):
        list1 = [make_result("shared"), make_result("unique1")]
        list2 = [make_result("shared"), make_result("unique2")]
        fused = reciprocal_rank_fusion([list1, list2], top_k=3)
        assert fused[0]["chunk_id"] == "shared"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_top_k_respected(self):
        results = [make_result(str(i)) for i in range(20)]
        fused = reciprocal_rank_fusion([results], top_k=5)
        assert len(fused) == 5
