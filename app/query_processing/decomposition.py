"""
Query decomposition: breaks complex questions into sub-questions,
retrieves evidence for each, then synthesizes a final answer.

Risk: over-decomposition can lose context. Implementation keeps
sub-questions targeted and tracks which evidence maps to which sub-question.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """\
Break the following complex question into {n} focused sub-questions that together cover the full answer.
Each sub-question should be self-contained and answerable independently.
Return ONLY the sub-questions, numbered 1. 2. 3.

Question: {query}

Sub-questions:"""


def decompose_query(query: str, llm_fn: Callable, n: int = 3) -> list[str]:
    """Break a complex question into sub-questions."""
    if not query.strip():
        return [query]
    prompt = DECOMPOSE_PROMPT.format(n=n, query=query.strip())
    try:
        result = llm_fn(prompt)
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        sub_questions = []
        for line in lines:
            if line[0].isdigit() and ". " in line:
                sub_questions.append(line.split(". ", 1)[1].strip())
            elif len(line) > 5:
                sub_questions.append(line)
        sub_questions = [q for q in sub_questions if len(q) > 5][:n]
        if sub_questions:
            logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
            return sub_questions
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
    return [query]


def decomposed_retrieve(
    query: str,
    llm_fn: Callable,
    retrieve_fn: Callable[[str], list[dict[str, Any]]],
    n_sub: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Decompose query and retrieve evidence for each sub-question.

    Returns:
        (all_evidence, sub_question_mapping) where sub_question_mapping
        maps each sub-question to its retrieved chunk_ids.
    """
    sub_questions = decompose_query(query, llm_fn, n=n_sub)
    all_evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    mapping: list[dict[str, str]] = []

    for sq in sub_questions:
        results = retrieve_fn(sq)
        sq_ids = []
        for r in results:
            cid = r.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_evidence.append(r)
            sq_ids.append(cid)
        mapping.append({"sub_question": sq, "chunk_ids": sq_ids})

    logger.info(
        f"Decomposed retrieve: {len(sub_questions)} sub-questions → "
        f"{len(all_evidence)} unique evidence chunks"
    )
    return all_evidence, mapping
