"""
Citation validation and formatting.
Ensures citations reference actual retrieved sources.
Removes or flags invalid citations.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def format_context(
    results: list[dict[str, Any]],
    max_chars_per_result: int = 800,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Format retrieved results as numbered context for the prompt.

    Returns:
        (context_string, source_list) where source_list contains
        metadata for citation validation.
    """
    parts = []
    sources = []
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        source = meta.get("source", r.get("source", "Unknown"))
        page = meta.get("page_number", r.get("page_number", "?"))
        text = r.get("text", "")[:max_chars_per_result]
        score = r.get("score", 0.0)

        parts.append(
            f"[{i}] Source: {source}, Page: {page}\n"
            f"    (Relevance: {score:.3f})\n"
            f"    {text}"
        )
        sources.append({
            "citation_num": i,
            "source": source,
            "page_number": page,
            "chunk_id": r.get("chunk_id", ""),
            "score": score,
        })

    return "\n\n".join(parts), sources


def validate_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """
    Validate that citations in the answer match retrieved sources.

    Returns:
        (validated_answer, list_of_warnings)

    Invalid citation numbers are flagged with [INVALID] markers.
    """
    valid_nums = {s["citation_num"] for s in sources}
    cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
    warnings = []

    invalid = cited_nums - valid_nums
    for num in sorted(invalid):
        answer = answer.replace(f"[{num}]", f"[{num}:INVALID]")
        warnings.append(f"Citation [{num}] does not correspond to a retrieved source")

    if warnings:
        logger.warning(f"Citation validation: {len(warnings)} invalid citations")

    return answer, warnings


def format_source_list(sources: list[dict[str, Any]]) -> str:
    """Format sources as a readable reference list."""
    if not sources:
        return ""
    lines = ["\n\n**Sources:**"]
    for s in sources:
        lines.append(
            f"[{s['citation_num']}] {s['source']}, page {s['page_number']}"
        )
    return "\n".join(lines)
