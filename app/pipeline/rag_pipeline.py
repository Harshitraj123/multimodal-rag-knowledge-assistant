"""
End-to-end RAG query pipeline.
Handles query processing, retrieval, and answer generation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

QUERY_STRATEGIES = {
    "none", "rewrite", "expand", "multi_query", "decompose", "hyde"
}


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    citation_warnings: list[str] = field(default_factory=list)
    retrieval_results: list[dict[str, Any]] = field(default_factory=list)
    visual_results: list[dict[str, Any]] = field(default_factory=list)
    query_used: str = ""
    original_query: str = ""
    query_strategy: str = "none"
    retrieval_method: str = "hybrid"
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0
    evidence_sufficient: bool = True


class RAGPipeline:
    """
    Orchestrates query enhancement → retrieval → visual retrieval → generation.
    Supports configurable query strategies and retrieval methods.
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        hybrid_retriever,
        visual_retriever,
        answer_generator,
        embedder_model: str,
        llm_provider=None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_enabled: bool = False,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.5,
        top_k: int = 5,
        candidate_k: int = 20,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.hybrid = hybrid_retriever
        self.visual = visual_retriever
        self.generator = answer_generator
        self.embedder_model = embedder_model
        self.llm = llm_provider
        self.reranker_model = reranker_model
        self.reranker_enabled = reranker_enabled
        self.mmr_enabled = mmr_enabled
        self.mmr_lambda = mmr_lambda
        self.top_k = top_k
        self.candidate_k = candidate_k

    def query(
        self,
        question: str,
        retrieval_method: str = "hybrid",
        query_strategy: str = "none",
        conversation_history: Optional[list[dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            question: User question.
            retrieval_method: "dense" | "sparse" | "hybrid".
            query_strategy: "none" | "rewrite" | "expand" | "multi_query" | "decompose" | "hyde".
            conversation_history: Prior conversation turns.
            top_k: Override default top_k.
        """
        t_start = time.time()
        top_k = top_k if top_k is not None else self.top_k
        conversation_history = conversation_history or []

        # Reformulate for conversation context
        if conversation_history and self.llm and self.llm.available:
            question_for_retrieval = self.generator.reformulate_for_history(
                question, conversation_history
            )
        else:
            question_for_retrieval = question

        # Apply query strategy
        question_for_retrieval = self._apply_query_strategy(
            question_for_retrieval, query_strategy
        )

        # Retrieve
        t_ret = time.time()
        results = self._retrieve(
            question_for_retrieval,
            method=retrieval_method,
            top_k=top_k,
        )
        retrieval_latency = time.time() - t_ret

        # Apply MMR if enabled
        if self.mmr_enabled and results:
            results = self._apply_mmr(question_for_retrieval, results, top_k)

        # Apply reranking if enabled
        if self.reranker_enabled and results:
            try:
                from app.retrieval.reranker import rerank
                results = rerank(
                    question_for_retrieval, results,
                    model_name=self.reranker_model,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning(f"Reranking failed, falling back: {e}")

        # Visual retrieval
        visual_results = []
        if self.visual:
            try:
                visual_results = self.visual.retrieve(question_for_retrieval, top_k=2)
            except Exception as e:
                logger.debug(f"Visual retrieval failed: {e}")

        # Generate answer
        t_gen = time.time()
        generated = self.generator.generate(
            question,
            results,
            visual_results=visual_results,
            conversation_history=conversation_history,
        )
        generation_latency = time.time() - t_gen

        total = time.time() - t_start
        return RAGResponse(
            answer=generated.answer,
            sources=generated.sources,
            citation_warnings=generated.citation_warnings,
            retrieval_results=results,
            visual_results=visual_results,
            query_used=question_for_retrieval,
            original_query=question,
            query_strategy=query_strategy,
            retrieval_method=retrieval_method,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=total,
            evidence_sufficient=generated.evidence_sufficient,
        )

    def _retrieve(self, query: str, method: str, top_k: int) -> list[dict[str, Any]]:
        """Dispatch to the appropriate retriever."""
        if method == "dense":
            return self.dense.retrieve(query, top_k=top_k)
        elif method == "sparse":
            return self.sparse.retrieve(query, top_k=top_k)
        else:  # hybrid (default)
            return self.hybrid.retrieve(query, top_k=top_k, candidate_k=self.candidate_k)

    def _apply_query_strategy(self, query: str, strategy: str) -> str:
        """Apply query transformation strategy."""
        if strategy == "none" or not self.llm or not self.llm.available:
            return query

        try:
            if strategy == "rewrite":
                from app.query_processing.rewrite import rewrite_query
                return rewrite_query(query, self.llm)
            elif strategy == "expand":
                from app.query_processing.expansion import expand_query
                return expand_query(query, self.llm)
            elif strategy == "multi_query":
                # Multi-query: generate variants, retrieve for each, return original
                # query for retrieval (merging happens in multi_query_retrieve)
                from app.query_processing.multi_query import generate_queries
                variants = generate_queries(query, self.llm, n=3)
                if variants:
                    # Return expanded query string combining original + variants
                    return " ".join([query] + variants[:2])
                return query
            elif strategy == "decompose":
                # Decompose: return original query; retrieval uses sub-questions
                from app.query_processing.decomposition import decompose_query
                sub_questions = decompose_query(query, self.llm, n=3)
                if sub_questions:
                    return " ".join(sub_questions)
                return query
            elif strategy == "hyde":
                # HyDE modifies the retrieval text but we log the original
                from app.query_processing.hyde import generate_hypothetical_document
                return generate_hypothetical_document(query, self.llm)
        except Exception as e:
            logger.warning(f"Query strategy '{strategy}' failed: {e}")
        return query

    def _apply_mmr(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """Apply Maximum Marginal Relevance diversification."""
        try:
            from app.embeddings.embedder import embed_query, embed_texts
            from app.retrieval.mmr import apply_mmr
            q_emb = embed_query(query, model_name=self.embedder_model)
            texts = [r["text"] for r in results]
            embs = embed_texts(texts, model_name=self.embedder_model)
            return apply_mmr(q_emb, results, embs, top_k=top_k, lambda_param=self.mmr_lambda)
        except Exception as e:
            logger.warning(f"MMR failed, skipping: {e}")
            return results
