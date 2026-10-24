"""
Document management service.
Provides high-level operations used by the API and frontend.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Coordinates document upload, ingestion, and management.
    Used by API routes and Streamlit frontend.
    """

    def __init__(
        self,
        ingestion_pipeline,
        document_store,
        upload_dir: str,
        max_upload_bytes: int,
    ):
        self.pipeline = ingestion_pipeline
        self.document_store = document_store
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = max_upload_bytes

    ALLOWED_EXTENSIONS = {
        ".pdf", ".docx", ".txt", ".md", ".csv", ".pptx",
        ".png", ".jpg", ".jpeg",
    }

    def save_upload(self, filename: str, content: bytes) -> Path:
        """
        Safely save an uploaded file.
        Validates extension, size, and sanitizes filename.
        Returns path to saved file.
        """
        if len(content) > self.max_upload_bytes:
            raise ValueError(
                f"File too large: {len(content)/1024/1024:.1f}MB > "
                f"{self.max_upload_bytes/1024/1024:.0f}MB limit"
            )
        # Sanitize filename — strip leading dots to prevent hidden file creation
        safe_name = Path(filename).name
        safe_name = "".join(
            c if c.isalnum() or c in (".", "-", "_") else "_"
            for c in safe_name
        )
        safe_name = safe_name.lstrip(".")  # prevent hidden files like ..etc
        if not safe_name:
            safe_name = "upload"
        ext = Path(safe_name).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type '{ext}' not supported. "
                f"Allowed: {sorted(self.ALLOWED_EXTENSIONS)}"
            )
        # Avoid path traversal
        dest = self.upload_dir / safe_name
        dest.write_bytes(content)
        logger.info(f"Saved upload: {dest.name} ({len(content)} bytes)")
        return dest

    def ingest_file(self, file_path: Path) -> dict[str, Any]:
        """Ingest a file and return result dict."""
        result = self.pipeline.ingest(file_path)
        return {
            "doc_id": result.doc_id,
            "file_name": result.file_name,
            "status": result.status,
            "chunks_added": result.chunks_added,
            "tables_indexed": result.tables_indexed,
            "visuals_indexed": result.visuals_indexed,
            "extraction_errors": result.extraction_errors,
            "processing_time_s": result.processing_time_s,
            "error_message": result.error_message,  # Fix: expose error to UI
        }

    def list_documents(self) -> list[dict[str, Any]]:
        """List all ingested documents."""
        return self.document_store.list_all()

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from all indexes."""
        return self.pipeline.delete_document(doc_id)

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Get document metadata by ID."""
        return self.document_store.get(doc_id)
