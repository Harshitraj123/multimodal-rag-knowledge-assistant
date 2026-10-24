"""Document upload, ingestion, listing, and deletion endpoints."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    UploadResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the server's upload directory.
    Does not ingest — call /ingest separately.
    """
    from app.api.dependencies import get_document_service
    svc = get_document_service()
    content = await file.read()
    try:
        saved_path = svc.save_upload(file.filename or "unknown", content)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return UploadResponse(
        file_name=saved_path.name,
        saved_path=str(saved_path),
        size_bytes=len(content),
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(request: IngestRequest):
    """Ingest a previously uploaded document into the vector store."""
    from app.api.dependencies import get_document_service
    svc = get_document_service()
    path = Path(request.file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}",
        )
    result = svc.ingest_file(path)
    return IngestResponse(**result)


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all ingested documents."""
    from app.api.dependencies import get_document_service
    svc = get_document_service()
    docs = svc.list_documents()
    doc_infos = [
        DocumentInfo(
            doc_id=d.get("doc_id", ""),
            file_name=d.get("file_name", ""),
            file_type=d.get("file_type", ""),
            chunk_count=d.get("chunk_count", 0),
            table_count=d.get("table_count", 0),
            visual_count=d.get("visual_count", 0),
            ingested_at=d.get("ingested_at"),
        )
        for d in docs
    ]
    return DocumentListResponse(documents=doc_infos, total=len(doc_infos))


@router.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document from all indexes."""
    from app.api.dependencies import get_document_service
    svc = get_document_service()
    if not svc.get_document(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {doc_id}",
        )
    deleted = svc.delete_document(doc_id)
    return DeleteResponse(
        doc_id=doc_id,
        deleted=deleted,
        message="Document deleted" if deleted else "Deletion failed",
    )
