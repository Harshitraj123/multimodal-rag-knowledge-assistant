"""
Query rewriting: clarifies and reformulates the user's question
without inventing facts. Uses the configured LLM.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """\
You are a query optimization assistant.
Rewrite the following search query to be clearer and more effective for document retrieval.
- Fix grammar and ambiguities
- Expand abbreviations if obvious
- Do NOT add facts not present in the original query
- Return ONLY the rewritten query, no explanation

Original query: {query}

Rewritten query:"""


def rewrite_query(query: str, llm_fn) -> str:
    """
    Rewrite a query using the provided LLM function.

    Args:
        query: Original user query.
        llm_fn: Callable(prompt: str) -> str.

    Returns:
        Rewritten query string. Falls back to original on failure.
    """
    if not query.strip():
        return query
    prompt = REWRITE_PROMPT.format(query=query.strip())
    try:
        result = llm_fn(prompt)
        rewritten = result.strip()
        if rewritten and len(rewritten) > 5:
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
    return query
