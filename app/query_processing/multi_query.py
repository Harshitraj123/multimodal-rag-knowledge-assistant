"""
Multi-query retrieval: generates multiple reformulations of a question,
retrieves results for each, and merges using deduplication.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = """\
Generate {n} different search queries that would help answer the following question.
Each query should approach the question from a different angle.
Return ONLY the queries, one per line, numbered 1. 2. 3.

Question: {query}

Queries:"""


def generate_queries(query: str, llm_fn: Callable, n: int = 3) -> list[str]:
    """Generate n alternative query formulations."""
    prompt = MULTI_QUERY_PROMPT.format(n=n, query=query.strip())
    try:
        result = llm_fn(prompt)
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        # Remove numbering
        queries = []
        for line in lines:
            if line and line[0].isdigit() and ". " in line:
                queries.append(line.split(". ", 1)[1].strip())
            elif line and len(line) > 5:
                queries.append(line)
        queries = [q for q in queries if len(q) > 5][:n]
        if queries:
            logger.info(f"Generated {len(queries)} query variants")
            return queries
    except Exception as e:
        logger.warning(f"Multi-query generation failed: {e}")
    return [query]


def multi_query_retrieve(
    query: str,
    llm_fn: Callable,
    retrieve_fn: Callable[[str], list[dict[str, Any]]],
    n_queries: int = 3,
    top_k_per_query: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate multiple queries, retrieve for each, merge and deduplicate.

    Returns merged, deduplicated list of results.
    Original query is always included.
    """
    queries = [query] + generate_queries(query, llm_fn, n=n_queries)
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []

    for q in queries:
        results = retrieve_fn(q)
        for r in results:
            cid = r.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(r)

    logger.info(f"Multi-query: {len(queries)} queries → {len(merged)} unique results")
    return merged
