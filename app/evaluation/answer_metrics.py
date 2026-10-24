"""
Answer quality metrics.
Distinguishes deterministic metrics from model-judged ones.
LLM-as-judge is clearly labelled and its limitations documented.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def citation_precision(
    answer: str,
    valid_citation_nums: set[int],
) -> float:
    """
    Fraction of citations in the answer that are valid.
    Deterministic metric.
    """
    cited = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
    if not cited:
        return 1.0  # no citations = no invalid citations
    valid_cited = cited & valid_citation_nums
    return len(valid_cited) / len(cited)


def answer_contains_evidence(
    answer: str,
    required_evidence: str,
    case_insensitive: bool = True,
) -> bool:
    """
    Check if the answer contains required evidence strings.
    Deterministic string-match metric.
    """
    if not required_evidence:
        return True
    ans = answer.lower() if case_insensitive else answer
    req = required_evidence.lower() if case_insensitive else required_evidence
    # Check for key phrases (split by comma or semicolon)
    phrases = [p.strip() for p in re.split(r"[,;]", req) if p.strip()]
    return any(phrase in ans for phrase in phrases) if phrases else True


def faithfulness_heuristic(
    answer: str,
    retrieved_texts: list[str],
) -> float:
    """
    Heuristic faithfulness: fraction of answer sentences that
    overlap with retrieved content.

    NOTE: This is a proxy metric. True faithfulness requires LLM evaluation.
    """
    if not retrieved_texts or not answer:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0

    combined_text = " ".join(retrieved_texts).lower()
    grounded = 0
    for sentence in sentences:
        words = set(re.findall(r"\b\w{4,}\b", sentence.lower()))
        if not words:
            continue
        overlap = sum(1 for w in words if w in combined_text)
        if overlap / len(words) >= 0.4:
            grounded += 1

    return grounded / len(sentences) if sentences else 0.0


def compute_answer_metrics(
    answer: str,
    sources: list[dict[str, Any]],
    required_evidence: str,
    retrieved_texts: list[str],
) -> dict[str, Any]:
    """Compute all answer quality metrics for a single response."""
    valid_nums = {s["citation_num"] for s in sources}
    return {
        "citation_precision": citation_precision(answer, valid_nums),
        "evidence_present": answer_contains_evidence(answer, required_evidence),
        "faithfulness_heuristic": faithfulness_heuristic(answer, retrieved_texts),
        "answer_length": len(answer),
        "citations_count": len(re.findall(r"\[\d+\]", answer)),
        "note": "faithfulness_heuristic is a proxy; not equivalent to LLM-judged faithfulness",
    }
