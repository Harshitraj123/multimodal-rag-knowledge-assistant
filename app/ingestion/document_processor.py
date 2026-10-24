"""
High-level document processing pipeline.
Coordinates loading, chunking, and metadata enrichment.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from .loaders import load_document, RawDocument

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """A document ready for indexing."""
    doc_id: str
    file_name: str
    file_type: str
    source_path: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_errors: list[str] = field(default_factory=list)
    processing_time_s: float = 0.0


def compute_doc_id(file_path: str | Path) -> str:
    """Stable document ID based on file content hash."""
    path = Path(file_path)
    h = hashlib.sha256()
    h.update(path.name.encode())
    try:
        # Use first 64KB for speed on large files
        with open(path, "rb") as f:
            h.update(f.read(65536))
    except Exception:
        pass
    return h.hexdigest()[:16]


def process_document(
    file_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> ProcessedDocument:
    """
    Full processing pipeline:
    1. Load raw content
    2. Chunk text pages
    3. Attach metadata to each chunk
    4. Return ProcessedDocument
    """
    start = time.time()
    path = Path(file_path)
    doc_id = compute_doc_id(path)

    raw: RawDocument = load_document(path)

    processed = ProcessedDocument(
        doc_id=doc_id,
        file_name=raw.file_name,
        file_type=raw.file_type,
        source_path=raw.source_path,
        tables=raw.tables,
        images=raw.images,
        metadata=raw.metadata,
        extraction_errors=raw.extraction_errors,
    )

    # Chunk text pages
    from app.chunking.splitter import chunk_text
    for page in raw.pages:
        text = page.get("text", "").strip()
        if not text:
            continue
        page_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk_text_val in enumerate(page_chunks):
            processed.chunks.append({
                "chunk_id": f"{doc_id}_p{page['page_number']}_c{i}",
                "doc_id": doc_id,
                "text": chunk_text_val,
                "source": raw.file_name,
                "page_number": page["page_number"],
                "chunk_index": i,
                "file_type": raw.file_type,
            })

    processed.processing_time_s = time.time() - start
    logger.info(
        f"Processed {path.name}: {len(processed.chunks)} chunks, "
        f"{len(processed.tables)} tables, {len(processed.images)} images "
        f"in {processed.processing_time_s:.2f}s"
    )
    return processed
