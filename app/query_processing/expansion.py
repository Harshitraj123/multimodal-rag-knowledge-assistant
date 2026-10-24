"""
Query expansion: adds related terms and synonyms to improve recall.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """\
Expand the following search query by adding related terms, synonyms, and alternative phrasings.
Return ONLY the expanded query (original + additions) as a single string.
Do not add unrelated topics.

Original query: {query}

Expanded query:"""


def expand_query(query: str, llm_fn) -> str:
    """
    Expand query with related terminology.
    Falls back to original query on failure.
    """
    if not query.strip():
        return query
    prompt = EXPANSION_PROMPT.format(query=query.strip())
    try:
        result = llm_fn(prompt)
        expanded = result.strip()
        if expanded and len(expanded) > len(query):
            logger.info(f"Query expanded: '{query}' → '{expanded[:100]}...'")
            return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
    return query
