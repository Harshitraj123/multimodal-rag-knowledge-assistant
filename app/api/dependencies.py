from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Singletons
_vector_store = None
_document_store = None
_bm25_retriever = None
_visual_retriever = None
_ingestion_pipeline = None
_rag_pipeline = None
_document_service = None

# Thread lock and initialization flag
_init_lock = threading.Lock()
_is_initialized = False

def _init_components():
    global _vector_store, _document_store, _bm25_retriever, _visual_retriever
    global _ingestion_pipeline, _rag_pipeline, _document_service
    global _is_initialized

    # Fast path: skip lock if already initialized
    if _is_initialized:
        return

    # Slow path: acquire lock for first-time setup
    with _init_lock:
        # Double-check inside lock to prevent race conditions
        if _is_initialized:
            return

        from app.config.settings import settings
        from app.config.logging_config import setup_logging
        setup_logging(settings.log_level)

        # Storage
        from app.storage.vector_store import VectorStore
        from app.storage.document_store import DocumentStore
        _vector_store = VectorStore(str(settings.chroma_dir_path))
        _document_store = DocumentStore(str(settings.chroma_dir_path / "documents.json"))

        # BM25
        from app.retrieval.sparse import BM25Retriever
        _bm25_retriever = BM25Retriever()
        bm25_manifest = settings.chroma_dir_path / "bm25_manifest.json"

        if bm25_manifest.exists():
            from app.storage.manifests import load_bm25_manifest
            corpus, ids, metas = load_bm25_manifest(bm25_manifest)
            if corpus:
                _bm25_retriever.build(corpus, ids, metas)

        # Visual retriever
        from app.retrieval.visual_retriever import VisualRetriever
        _visual_retriever = VisualRetriever(
            str(settings.chroma_dir_path),
            settings.embedding_model,
        )

        # Retrievers
        from app.retrieval.dense import DenseRetriever
        from app.retrieval.hybrid import HybridRetriever
        dense = DenseRetriever(_vector_store, settings.embedding_model)
        hybrid = HybridRetriever(dense, _bm25_retriever)

        # LLM providers
        from app.generation.llm_provider import LLMProvider
        from app.generation.vision_provider import VisionProvider
        llm = LLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            provider=settings.llm_provider,
        )
        vision = VisionProvider(
            provider=settings.vision_provider,
            api_key=settings.vision_api_key,
            model=settings.vision_model,
            base_url=settings.ollama_base_url,
        )

        if not llm.available:
            logger.warning("LLM provider not configured — running in retrieval-only mode")
        if not vision.available:
            logger.warning("Vision provider not configured — visual analysis disabled")

        # Answer generator
        from app.generation.answer_generator import AnswerGenerator
        generator = AnswerGenerator(llm, vision if vision.available else None)

        # Ingestion pipeline
        from app.pipeline.ingestion_pipeline import IngestionPipeline
        _ingestion_pipeline = IngestionPipeline(
            vector_store=_vector_store,
            document_store=_document_store,
            bm25_retriever=_bm25_retriever,
            visual_retriever=_visual_retriever,
            embedder_model=settings.embedding_model,
            vision_provider=vision if vision.available else None,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            bm25_manifest_path=str(bm25_manifest),
        )

        # RAG pipeline
        from app.pipeline.rag_pipeline import RAGPipeline
        _rag_pipeline = RAGPipeline(
            dense_retriever=dense,
            sparse_retriever=_bm25_retriever,
            hybrid_retriever=hybrid,
            visual_retriever=_visual_retriever,
            answer_generator=generator,
            embedder_model=settings.embedding_model,
            llm_provider=llm,
            reranker_model=settings.reranker_model,
            reranker_enabled=settings.reranker_enabled,
            mmr_enabled=settings.mmr_enabled,
            mmr_lambda=settings.mmr_lambda,
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
        )

        # Document service
        from app.services.document_service import DocumentService
        _document_service = DocumentService(
            ingestion_pipeline=_ingestion_pipeline,
            document_store=_document_store,
            upload_dir=str(settings.upload_dir_path),
            max_upload_bytes=settings.max_upload_bytes,
        )

        # Mark as completely initialized ONLY if everything succeeds
        _is_initialized = True
        logger.info("All application components initialized")


def get_rag_pipeline():
    _init_components()
    return _rag_pipeline


def get_document_service():
    _init_components()
    return _document_service


def get_settings():
    from app.config.settings import settings
    return settings