"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from app.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check application health and capability status.
    Returns configuration summary without revealing secrets.
    """
    from app.config.settings import settings

    caps = settings.report_capabilities()

    # Get index stats
    try:
        from app.api.dependencies import _vector_store, _document_store
        chunks = _vector_store.count() if _vector_store else 0
        docs = _document_store.count() if _document_store else 0
    except Exception:
        chunks = 0
        docs = 0

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        capabilities=caps,
        indexed_documents=docs,
        indexed_chunks=chunks,
    )
