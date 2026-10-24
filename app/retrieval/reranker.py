"""
Cross-encoder reranker for second-stage relevance scoring.

Two-stage retrieval:
  Stage 1: Retrieve broad candidates (dense/sparse/hybrid) — fast but approximate.
  Stage 2: Rerank candidates with a cross-encoder — slower but more accurate.

Cross-encoders jointly encode query + passage and produce a calibrated
relevance score, making them more accurate than bi-encoder similarity
at the cost of being O(n) per query.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_reranker_cache: dict[str, object] = {}


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if model_name not in _reranker_cache:
        logger.info(f"Loading reranker model: {model_name}")
        try:
            from sentence_transformers import CrossEncoder
            _reranker_cache[model_name] = CrossEncoder(model_name)
            logger.info("Reranker loaded")
        except ImportError:
            raise ImportError("sentence-transformers not installed for CrossEncoder")
        except Exception as e:
            raise RuntimeError(f"Failed to load reranker '{model_name}': {e}") from e
    return _reranker_cache[model_name]


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank candidates using cross-encoder scores.

    Args:
        query: Original query string.
        candidates: Candidate results with 'text' field.
        model_name: Cross-encoder model identifier.
        top_k: Number of results to return after reranking.

    Returns:
        Top-k results sorted by reranker score (descending),
        with 'reranker_score', 'original_rank', and updated 'retrieval_method'.
    """
    if not candidates:
        return []

    t = time.time()
    reranker = get_reranker(model_name)

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []
    for new_rank, (candidate, score) in enumerate(ranked, start=1):
        r = dict(candidate)
        r["reranker_score"] = float(score)
        r["score"] = float(score)
        r["reranker_rank"] = new_rank
        r["retrieval_method"] = r.get("retrieval_method", "dense") + "+reranked"
        results.append(r)

    elapsed = time.time() - t
    logger.info(f"Reranked {len(candidates)} → {len(results)} in {elapsed:.3f}s")
    return results
