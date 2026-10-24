"""Unit tests for Maximum Marginal Relevance."""
import unittest
try:
    import pytest
except ImportError:
    pytest = None
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.retrieval.mmr import apply_mmr


def make_embedding(val: float, dim: int = 4) -> list[float]:
    return [val] * dim


def make_result(chunk_id: str, text: str = "text") -> dict:
    return {"chunk_id": chunk_id, "text": text, "score": 0.9, "metadata": {}}


class TestMMR(unittest.TestCase):
    def test_returns_correct_number(self):
        query_emb = make_embedding(1.0)
        candidates = [make_result(f"doc{i}") for i in range(10)]
        embs = [make_embedding(float(i) * 0.1) for i in range(10)]
        result = apply_mmr(query_emb, candidates, embs, top_k=3)
        assert len(result) == 3

    def test_empty_candidates(self):
        result = apply_mmr(make_embedding(1.0), [], [], top_k=5)
        assert result == []

    def test_mmr_rank_added(self):
        query_emb = make_embedding(1.0)
        candidates = [make_result(f"doc{i}") for i in range(5)]
        embs = [make_embedding(float(i) * 0.2) for i in range(5)]
        result = apply_mmr(query_emb, candidates, embs, top_k=3)
        assert all("mmr_rank" in r for r in result)

    def test_lambda_1_keeps_most_relevant(self):
        """With lambda=1.0, MMR becomes pure relevance ranking."""
        query_emb = [1.0, 0.0, 0.0, 0.0]
        # First candidate is most similar to query
        candidates = [make_result("best"), make_result("ok"), make_result("worst")]
        embs = [
            [1.0, 0.0, 0.0, 0.0],  # identical to query
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        result = apply_mmr(query_emb, candidates, embs, top_k=1, lambda_param=1.0)
        assert result[0]["chunk_id"] == "best"
