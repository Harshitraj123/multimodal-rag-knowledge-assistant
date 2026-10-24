"""
Hypothetical Document Embeddings (HyDE).

How it works:
1. Generate a hypothetical answer document using the LLM (no retrieval yet).
2. Embed the hypothetical document (not the query).
3. Use that embedding to retrieve real documents.
4. Generate the final answer from REAL retrieved evidence (not the hypothesis).

IMPORTANT: The hypothetical document is a retrieval aid only.
It is never presented as an answer or treated as verified evidence.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

HYDE_PROMPT = """\
Write a short, factual paragraph that would be a good answer to the following question.
This is used only for improving document retrieval — it does not need to be perfectly accurate.
Keep it concise (2-4 sentences).

Question: {query}

Hypothetical answer:"""


def generate_hypothetical_document(query: str, llm_fn: Callable) -> str:
    """Generate a hypothetical answer for embedding-based retrieval."""
    prompt = HYDE_PROMPT.format(query=query.strip())
    try:
        hypothesis = llm_fn(prompt).strip()
        logger.info(f"HyDE hypothesis generated ({len(hypothesis)} chars)")
        return hypothesis if hypothesis else query
    except Exception as e:
        logger.warning(f"HyDE hypothesis generation failed: {e}")
        return query


def hyde_retrieve(
    query: str,
    llm_fn: Callable,
    retrieve_fn: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Retrieve using HyDE: embed a hypothetical answer, retrieve real documents.

    The hypothetical document is used only for retrieval — never as evidence.
    Final answer must be generated from retrieved real documents.
    """
    hypothesis = generate_hypothetical_document(query, llm_fn)
    # Retrieve using the hypothesis as the query text
    results = retrieve_fn(hypothesis)
    # Tag results to make the HyDE provenance visible
    for r in results:
        r["hyde_hypothesis_used"] = True
        r["retrieval_method"] = r.get("retrieval_method", "dense") + "+hyde"
    logger.info(f"HyDE retrieved {len(results)} results")
    return results
