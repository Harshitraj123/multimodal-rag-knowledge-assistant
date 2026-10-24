"""
Grounded answer generation pipeline.
Combines retrieved evidence with LLM generation and citation validation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_ANSWER_PROMPT,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    CONVERSATIONAL_REFORMAT_PROMPT,
    DECOMPOSED_SYNTHESIS_PROMPT,
)
from .citations import format_context, validate_citations, format_source_list

logger = logging.getLogger(__name__)

MIN_EVIDENCE_SCORE = 0.1  # minimum score to consider a result useful


@dataclass
class GeneratedAnswer:
    """Container for a generated answer with full provenance."""
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    citation_warnings: list[str] = field(default_factory=list)
    retrieval_results: list[dict[str, Any]] = field(default_factory=list)
    visual_results: list[dict[str, Any]] = field(default_factory=list)
    query_used: str = ""
    original_query: str = ""
    retrieval_method: str = ""
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0
    evidence_sufficient: bool = True


class AnswerGenerator:
    """
    Orchestrates retrieval and answer generation.
    Grounds responses in retrieved evidence with citation validation.
    """

    def __init__(self, llm_provider, vision_provider=None):
        self.llm = llm_provider
        self.vision = vision_provider

    def generate(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]],
        visual_results: Optional[list[dict[str, Any]]] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer from retrieved evidence.

        Args:
            query: The user's question.
            retrieved_results: Text/table chunks from retrieval.
            visual_results: Optional visual assets from visual retrieval.
            conversation_history: Optional list of {"role": ..., "content": ...} dicts.

        Returns:
            GeneratedAnswer with answer, citations, and metadata.
        """
        t_start = time.time()
        visual_results = visual_results or []

        # Filter very low-scoring results
        good_results = [r for r in retrieved_results if r.get("score", 0) >= MIN_EVIDENCE_SCORE]
        if not good_results and retrieved_results:
            good_results = retrieved_results[:3]  # use top-3 anyway

        if not good_results and not visual_results:
            return GeneratedAnswer(
                answer=INSUFFICIENT_EVIDENCE_RESPONSE,
                query_used=query,
                original_query=query,
                evidence_sufficient=False,
                total_latency_s=time.time() - t_start,
            )

        if not self.llm.available:
            # Retrieval-only mode
            context, sources = format_context(good_results)
            source_list = format_source_list(sources)
            answer = (
                f"**Retrieval-only mode** (LLM not configured).\n\n"
                f"Top retrieved evidence for: *{query}*\n\n"
                f"{context}{source_list}"
            )
            return GeneratedAnswer(
                answer=answer,
                sources=sources,
                retrieval_results=good_results,
                visual_results=visual_results,
                query_used=query,
                original_query=query,
                evidence_sufficient=True,
                total_latency_s=time.time() - t_start,
            )

        # Format context
        context, sources = format_context(good_results)

        # Add visual evidence if available and vision model configured
        visual_context = ""
        if visual_results and self.vision and self.vision.available:
            visual_parts = []
            for vr in visual_results[:2]:  # limit to 2 images per query
                try:
                    desc = self.vision.describe_image(
                        vr["image_bytes"],
                        question=query,
                        context=vr.get("description", ""),
                    )
                    src = vr.get("metadata", {}).get("source", "unknown")
                    pg = vr.get("metadata", {}).get("page_number", "?")
                    visual_parts.append(f"[Visual from {src}, page {pg}]: {desc}")
                except Exception as e:
                    logger.warning(f"Vision analysis failed: {e}")
            if visual_parts:
                visual_context = "\n\nVisual Evidence:\n" + "\n\n".join(visual_parts)

        full_context = context + visual_context

        # Build prompt
        prompt = RAG_ANSWER_PROMPT.format(
            question=query,
            context=full_context,
        )

        t_gen = time.time()
        try:
            raw_answer = self.llm.generate(
                prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raw_answer = f"Answer generation failed: {e}\n\n{INSUFFICIENT_EVIDENCE_RESPONSE}"

        gen_latency = time.time() - t_gen

        # Validate citations
        validated_answer, warnings = validate_citations(raw_answer, sources)
        source_list = format_source_list(sources)
        final_answer = validated_answer + source_list

        return GeneratedAnswer(
            answer=final_answer,
            sources=sources,
            citation_warnings=warnings,
            retrieval_results=good_results,
            visual_results=visual_results,
            query_used=query,
            original_query=query,
            retrieval_method=good_results[0].get("retrieval_method", "") if good_results else "",
            generation_latency_s=gen_latency,
            total_latency_s=time.time() - t_start,
            evidence_sufficient=True,
        )

    def reformulate_for_history(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """Reformulate a follow-up question given conversation history."""
        if not conversation_history or not self.llm.available:
            return query
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:200]}"
            for m in conversation_history[-6:]  # bounded history
        )
        prompt = CONVERSATIONAL_REFORMAT_PROMPT.format(
            history=history_str,
            question=query,
        )
        try:
            return self.llm.generate(prompt, max_tokens=128).strip() or query
        except Exception:
            return query
