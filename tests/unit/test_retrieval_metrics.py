"""Unit tests for retrieval evaluation metrics."""
import unittest
try:
    import pytest
except ImportError:
    pytest = None
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.evaluation.retrieval_metrics import (
    hit_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, compute_retrieval_metrics
)


def make_result(source: str, score: float = 0.9):
    return {
        "chunk_id": f"id_{source}",
        "text": f"Text from {source}",
        "metadata": {"source": source},
        "score": score,
    }


class TestHitAtK(unittest.TestCase):
    def test_hit_when_relevant_in_top_k(self):
        results = [make_result("doc_a.pdf"), make_result("doc_b.pdf")]
        assert hit_at_k(results, ["doc_a.pdf"], k=5) == 1.0

    def test_miss_when_relevant_not_in_top_k(self):
        results = [make_result("doc_b.pdf"), make_result("doc_c.pdf")]
        assert hit_at_k(results, ["doc_a.pdf"], k=5) == 0.0

    def test_cutoff_respected(self):
        results = [make_result("doc_b.pdf")] * 4 + [make_result("doc_a.pdf")]
        assert hit_at_k(results, ["doc_a.pdf"], k=3) == 0.0
        assert hit_at_k(results, ["doc_a.pdf"], k=5) == 1.0


class TestRecallAtK(unittest.TestCase):
    def test_full_recall(self):
        results = [make_result("doc_a.pdf"), make_result("doc_b.pdf")]
        assert recall_at_k(results, ["doc_a.pdf", "doc_b.pdf"], k=5) == 1.0

    def test_partial_recall(self):
        results = [make_result("doc_a.pdf")]
        assert recall_at_k(results, ["doc_a.pdf", "doc_b.pdf"], k=5) == 0.5

    def test_empty_expected_returns_zero(self):
        results = [make_result("doc_a.pdf")]
        assert recall_at_k(results, [], k=5) == 0.0


class TestReciprocalRank(unittest.TestCase):
    def test_first_result_relevant(self):
        results = [make_result("doc_a.pdf"), make_result("doc_b.pdf")]
        assert reciprocal_rank(results, ["doc_a.pdf"]) == 1.0

    def test_second_result_relevant(self):
        results = [make_result("doc_b.pdf"), make_result("doc_a.pdf")]
        assert reciprocal_rank(results, ["doc_a.pdf"]) == 0.5

    def test_no_relevant_result(self):
        results = [make_result("doc_b.pdf")]
        assert reciprocal_rank(results, ["doc_a.pdf"]) == 0.0


class TestNDCG(unittest.TestCase):
    def test_perfect_ranking(self):
        results = [make_result("doc_a.pdf"), make_result("doc_b.pdf")]
        score = ndcg_at_k(results, ["doc_a.pdf"], k=5)
        assert score == 1.0

    def test_irrelevant_results_give_zero(self):
        results = [make_result("doc_c.pdf"), make_result("doc_d.pdf")]
        score = ndcg_at_k(results, ["doc_a.pdf"], k=5)
        assert score == 0.0


class TestComputeMetrics(unittest.TestCase):
    def test_returns_all_metrics(self):
        results = [make_result("doc_a.pdf")]
        metrics = compute_retrieval_metrics(results, ["doc_a.pdf"], k=5)
        assert "hit@5" in metrics
        assert "recall@5" in metrics
        assert "mrr" in metrics
        assert "ndcg@5" in metrics
