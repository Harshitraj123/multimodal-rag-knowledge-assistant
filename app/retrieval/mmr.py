"""
Maximum Marginal Relevance (MMR) for result diversification.

MMR selects results that are relevant to the query but dissimilar to
already-selected results, controlled by lambda_param:
  - lambda_param = 1.0 → pure relevance (same as top-k)
  - lambda_param = 0.0 → pure diversity
  - lambda_param = 0.5 → balanced (recommended default)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def apply_mmr(
    query_embedding: list[float],
    candidates: list[dict[str, Any]],
    candidate_embeddings: list[list[float]],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Apply MMR to a set of candidate results.

    Args:
        query_embedding: Embedding of the original query.
        candidates: Candidate result dicts with at least 'chunk_id' and 'text'.
        candidate_embeddings: Embeddings corresponding to each candidate.
        top_k: Number of results to select.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        Selected results in MMR order, with 'mmr_score' field added.
    """
    if not candidates:
        return []
    if len(candidates) != len(candidate_embeddings):
        raise ValueError("candidates and candidate_embeddings must have same length")

    top_k = min(top_k, len(candidates))
    q = np.array(query_embedding, dtype=np.float32)
    embs = np.array(candidate_embeddings, dtype=np.float32)

    # Cosine similarities to query
    query_sims = _cosine_batch(q, embs)

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(top_k):
        if not remaining:
            break

        if not selected_indices:
            # First selection: pick most relevant to query
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            # MMR: relevance minus max similarity to already selected
            selected_embs = embs[selected_indices]
            scores = []
            for i in remaining:
                rel = query_sims[i]
                redundancy = float(np.max(_cosine_batch(embs[i], selected_embs)))
                mmr_score = lambda_param * rel - (1 - lambda_param) * redundancy
                scores.append((i, mmr_score))
            best = max(scores, key=lambda x: x[1])[0]

        selected_indices.append(best)
        remaining.remove(best)

    results = []
    for rank, idx in enumerate(selected_indices):
        r = dict(candidates[idx])
        r["mmr_score"] = float(query_sims[idx])
        r["mmr_rank"] = rank + 1
        r["retrieval_method"] = r.get("retrieval_method", "dense") + "+mmr"
        results.append(r)

    logger.info(f"MMR selected {len(results)} from {len(candidates)} candidates")
    return results


def _cosine_batch(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a vector and each row of a matrix."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = matrix / norms
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    return normed @ vec_norm
