"""
Dense semantic retrieval using vector similarity search.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Retrieves semantically similar chunks using embedding-based search.
    """

    def __init__(self, vector_store, embedder_model: str):
        self.vector_store = vector_store
        self.embedder_model = embedder_model

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Embed query and retrieve top_k similar chunks.
        Returns results sorted by descending similarity score.
        """
        from app.embeddings.embedder import embed_query
        t = time.time()
        query_emb = embed_query(query, model_name=self.embedder_model)
        results = self.vector_store.query(query_emb, top_k=top_k, where=where)
        elapsed = time.time() - t
        logger.info(f"Dense retrieval: {len(results)} results in {elapsed:.3f}s")
        for r in results:
            r["retrieval_method"] = "dense"
            r["retrieval_latency_s"] = elapsed
        return results
