"""
Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.

RRF formula: score(d) = sum_r 1 / (k + rank_r(d))
where k=60 is a smoothing constant and rank_r(d) is the rank of document d
in result list r (1-indexed).

RRF is parameter-free (beyond k), stable, and does not require score
normalization — making it appropriate for combining BM25 and dense scores
which are on incomparable scales.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the original RRF paper


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    top_k: int = 10,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked result lists using RRF.

    Args:
        result_lists: Each list is a ranked result from one retrieval method.
                      Items must have 'chunk_id' and 'text' fields.
        top_k: Number of final results to return.
        k: RRF smoothing constant (default 60).

    Returns:
        Fused and re-ranked list of results with 'rrf_score' field.
    """
    if not result_lists:
        return []

    # Accumulate RRF scores across all lists
    rrf_scores: dict[str, float] = {}
    best_result: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in best_result:
                best_result[chunk_id] = dict(result)

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused = []
    for cid in sorted_ids[:top_k]:
        r = dict(best_result[cid])
        r["rrf_score"] = rrf_scores[cid]
        r["score"] = rrf_scores[cid]
        r["retrieval_method"] = "hybrid_rrf"
        fused.append(r)

    logger.info(f"RRF fusion: {sum(len(l) for l in result_lists)} candidates → {len(fused)} fused results")
    return fused
