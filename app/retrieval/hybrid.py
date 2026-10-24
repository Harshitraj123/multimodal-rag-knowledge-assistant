"""
Hybrid retrieval: combines dense and sparse results via RRF.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Retrieves using both dense (semantic) and sparse (BM25) methods,
    then fuses results using Reciprocal Rank Fusion.
    """

    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve using both methods and fuse via RRF.

        candidate_k: how many results to fetch from each method before fusion.
        top_k: final number of results to return.
        """
        t = time.time()
        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse.retrieve(query, top_k=candidate_k)

        lists_to_fuse = []
        if dense_results:
            lists_to_fuse.append(dense_results)
        if sparse_results:
            lists_to_fuse.append(sparse_results)

        if not lists_to_fuse:
            return []

        fused = reciprocal_rank_fusion(lists_to_fuse, top_k=top_k)
        elapsed = time.time() - t
        logger.info(
            f"Hybrid retrieval: dense={len(dense_results)}, "
            f"sparse={len(sparse_results)}, fused={len(fused)} in {elapsed:.3f}s"
        )
        for r in fused:
            r["retrieval_latency_s"] = elapsed
        return fused
