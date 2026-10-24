"""Query and search endpoints."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceInfo,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Ask a question against the indexed document corpus.
    Returns grounded answer with citations and retrieval metadata.
    """
    from app.api.dependencies import get_rag_pipeline
    pipeline = get_rag_pipeline()

    try:
        response = pipeline.query(
            question=request.question,
            retrieval_method=request.retrieval_method,
            query_strategy=request.query_strategy,
            conversation_history=request.conversation_history,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {e}",
        )

    sources = [SourceInfo(**s) for s in response.sources]
    return QueryResponse(
        answer=response.answer,
        sources=sources,
        citation_warnings=response.citation_warnings,
        query_used=response.query_used,
        original_query=response.original_query,
        query_strategy=response.query_strategy,
        retrieval_method=response.retrieval_method,
        retrieval_latency_s=response.retrieval_latency_s,
        generation_latency_s=response.generation_latency_s,
        total_latency_s=response.total_latency_s,
        evidence_sufficient=response.evidence_sufficient,
        retrieval_results_count=len(response.retrieval_results),
    )


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    """
    Retrieve documents without generating an answer.
    Use for inspection and debugging of retrieval behavior.
    """
    from app.api.dependencies import get_rag_pipeline
    pipeline = get_rag_pipeline()

    t = time.time()
    try:
        results = pipeline._retrieve(
            request.query,
            method=request.retrieval_method,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        )
    elapsed = time.time() - t

    search_results = [
        SearchResult(
            chunk_id=r.get("chunk_id", ""),
            text=r.get("text", "")[:500],  # truncate for response
            source=r.get("metadata", {}).get("source", r.get("source", "")),
            page_number=r.get("metadata", {}).get("page_number", r.get("page_number", "?")),
            score=r.get("score", 0.0),
            retrieval_method=r.get("retrieval_method", ""),
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]
    return SearchResponse(
        query=request.query,
        results=search_results,
        retrieval_method=request.retrieval_method,
        latency_s=elapsed,
        total_results=len(results),
    )
