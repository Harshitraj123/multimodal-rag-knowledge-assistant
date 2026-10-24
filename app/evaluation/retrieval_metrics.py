"""
Retrieval evaluation metrics.
All metrics are deterministic (no LLM calls required).

Metrics implemented:
- Hit@k: Was any expected source retrieved in the top k?
- Recall@k: What fraction of expected sources were retrieved?
- MRR: Mean Reciprocal Rank of first relevant result
- nDCG@k: Normalized Discounted Cumulative Gain
"""
from __future__ import annotations

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_relevant(result: dict[str, Any], expected_sources: list[str]) -> bool:
    """Check if a retrieved result matches an expected source."""
    source = result.get("metadata", {}).get("source", result.get("source", ""))
    return any(exp.lower() in source.lower() or source.lower() in exp.lower()
               for exp in expected_sources)


def hit_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    k: int,
) -> float:
    """1.0 if any expected source appears in top-k results, else 0.0."""
    for r in results[:k]:
        if _is_relevant(r, expected_sources):
            return 1.0
    return 0.0


def recall_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    k: int,
) -> float:
    """Fraction of expected sources retrieved in top-k."""
    if not expected_sources:
        return 0.0
    retrieved = {
        result.get("metadata", {}).get("source", result.get("source", ""))
        for result in results[:k]
    }
    hits = sum(
        1 for exp in expected_sources
        if any(exp.lower() in r.lower() or r.lower() in exp.lower() for r in retrieved)
    )
    return hits / len(expected_sources)


def reciprocal_rank(
    results: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    """1/rank of the first relevant result, or 0 if none found."""
    for rank, r in enumerate(results, start=1):
        if _is_relevant(r, expected_sources):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    k: int,
) -> float:
    """nDCG@k with binary relevance labels."""
    def dcg(rels: list[float]) -> float:
        return sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(rels)
        )

    relevances = [
        1.0 if _is_relevant(r, expected_sources) else 0.0
        for r in results[:k]
    ]
    ideal = sorted(relevances, reverse=True)
    actual_dcg = dcg(relevances)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_retrieval_metrics(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    k: int = 5,
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query."""
    return {
        f"hit@{k}": hit_at_k(results, expected_sources, k),
        f"recall@{k}": recall_at_k(results, expected_sources, k),
        "mrr": reciprocal_rank(results, expected_sources),
        f"ndcg@{k}": ndcg_at_k(results, expected_sources, k),
        "num_retrieved": len(results),
    }


def aggregate_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across multiple evaluation cases."""
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in keys
    }
