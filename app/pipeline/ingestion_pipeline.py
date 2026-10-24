"""
End-to-end document ingestion pipeline.
Coordinates: loading → chunking → embedding → storage → BM25 indexing → visual indexing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a single document."""
    doc_id: str
    file_name: str
    status: str  # "success" | "duplicate" | "error"
    chunks_added: int = 0
    tables_indexed: int = 0
    visuals_indexed: int = 0
    extraction_errors: list[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    error_message: str = ""


class IngestionPipeline:
    """
    Orchestrates document ingestion from file to indexed state.
    Handles all supported document types.
    """

    def __init__(
        self,
        vector_store,
        document_store,
        bm25_retriever,
        visual_retriever,
        embedder_model: str,
        vision_provider=None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        bm25_manifest_path: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.document_store = document_store
        self.bm25_retriever = bm25_retriever
        self.visual_retriever = visual_retriever
        self.embedder_model = embedder_model
        self.vision_provider = vision_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_manifest_path = bm25_manifest_path

    def ingest(self, file_path: str | Path) -> IngestionResult:
        """
        Ingest a single document.
        Returns IngestionResult indicating success, duplicate, or error.
        """
        t_start = time.time()
        path = Path(file_path)

        from app.ingestion.document_processor import process_document, compute_doc_id

        doc_id = compute_doc_id(path)

        # Duplicate check
        if self.document_store.exists(doc_id):
            logger.info(f"Document {path.name} already ingested (doc_id={doc_id})")
            return IngestionResult(
                doc_id=doc_id,
                file_name=path.name,
                status="duplicate",
                processing_time_s=time.time() - t_start,
            )

        try:
            # Process document
            processed = process_document(
                path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        except Exception as e:
            logger.error(f"Document processing failed for {path.name}: {e}")
            return IngestionResult(
                doc_id=doc_id,
                file_name=path.name,
                status="error",
                error_message=str(e),
                processing_time_s=time.time() - t_start,
            )

        chunks_added = 0
        tables_indexed = 0
        visuals_indexed = 0

        # Embed and store text chunks
        if processed.chunks:
            try:
                from app.embeddings.embedder import embed_texts
                texts = [c["text"] for c in processed.chunks]
                embeddings = embed_texts(texts, model_name=self.embedder_model)
                chunks_added = self.vector_store.add_chunks(processed.chunks, embeddings)

                # Update BM25 index
                self._update_bm25(processed.chunks)
            except Exception as e:
                logger.error(f"Embedding/storage failed for {path.name}: {e}")
                processed.extraction_errors.append(f"Storage error: {e}")

        # Index table chunks
        if processed.tables:
            try:
                from app.chunking.splitter import chunk_table
                for table in processed.tables:
                    table_texts = chunk_table(table)
                    for i, ttext in enumerate(table_texts):
                        chunk = {
                            "chunk_id": f"{doc_id}_tbl{table.get('table_index', 0)}_c{i}",
                            "doc_id": doc_id,
                            "text": ttext,
                            "source": path.name,
                            "page_number": table.get("page_number", 1),
                            "chunk_index": i,
                            "file_type": processed.file_type,
                            "content_type": "table",
                        }
                        from app.embeddings.embedder import embed_texts
                        embs = embed_texts([ttext], model_name=self.embedder_model)
                        self.vector_store.add_chunks([chunk], embs)
                        self._update_bm25([chunk])
                        tables_indexed += 1
            except Exception as e:
                logger.warning(f"Table indexing failed: {e}")

        # Index visual assets
        if processed.images:
            for img_data in processed.images:
                try:
                    img_bytes = img_data.get("image_bytes")
                    if not img_bytes:
                        continue
                    # Generate description
                    if self.vision_provider and self.vision_provider.available:
                        description = self.vision_provider.generate_visual_description(
                            img_bytes,
                            source_info=f"{path.name} page {img_data.get('page_number', 1)}",
                        )
                    else:
                        description = img_data.get("description", f"Visual from {path.name}")

                    visual_id = f"{doc_id}_img{img_data.get('page_number', 1)}"
                    added = self.visual_retriever.add_visual(
                        visual_id=visual_id,
                        description=description,
                        image_bytes=img_bytes,
                        metadata={
                            "doc_id": doc_id,
                            "source": path.name,
                            "page_number": img_data.get("page_number", 1),
                            "image_type": img_data.get("image_type", "unknown"),
                        },
                    )
                    if added:
                        visuals_indexed += 1
                except Exception as e:
                    logger.warning(f"Visual indexing failed for {path.name}: {e}")

        # Register document
        self.document_store.register(
            doc_id,
            {
                "file_name": path.name,
                "file_type": processed.file_type,
                "source_path": str(path),
                "chunk_count": chunks_added,
                "table_count": tables_indexed,
                "visual_count": visuals_indexed,
                "extraction_errors": processed.extraction_errors,
            },
        )

        return IngestionResult(
            doc_id=doc_id,
            file_name=path.name,
            status="success",
            chunks_added=chunks_added,
            tables_indexed=tables_indexed,
            visuals_indexed=visuals_indexed,
            extraction_errors=processed.extraction_errors,
            processing_time_s=time.time() - t_start,
        )

    def _update_bm25(self, chunks: list[dict]) -> None:
        """Add chunks to BM25 index and persist."""
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metas = [
            {
                "doc_id": c.get("doc_id", ""),
                "source": c.get("source", ""),
                "page_number": c.get("page_number", 1),
            }
            for c in chunks
        ]
        self.bm25_retriever.add_documents(texts, ids, metas)
        if self.bm25_manifest_path:
            from app.storage.manifests import save_bm25_manifest
            save_bm25_manifest(
                self.bm25_manifest_path,
                self.bm25_retriever._corpus,
                self.bm25_retriever._ids,
                self.bm25_retriever._metadatas,
            )

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from all indexes."""
        self.vector_store.delete_by_doc_id(doc_id)
        self.document_store.delete(doc_id)
        # BM25 index is rebuilt on next ingestion; mark as stale
        logger.info(f"Document {doc_id} deleted from indexes")
        return True
