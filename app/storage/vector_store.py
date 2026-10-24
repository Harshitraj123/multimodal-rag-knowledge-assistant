"""
Chroma-based persistent vector store.
Handles document storage, retrieval, deletion, and metadata filtering.
Keeps storage logic isolated from business logic.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_documents"
VISUAL_COLLECTION_NAME = "rag_visuals"


class VectorStore:
    """
    Wraps ChromaDB for text chunk storage and retrieval.
    Uses cosine similarity for semantic search.
    """

    def __init__(self, persist_directory: str, collection_name: str = COLLECTION_NAME):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.info(f"ChromaDB client initialized at {self.persist_directory}")
            except ImportError:
                raise ImportError("chromadb not installed. Run: pip install chromadb")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(f"Collection '{self.collection_name}' ready ({self._collection.count()} docs)")
        return self._collection

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add text chunks with pre-computed embeddings.
        Returns number of chunks added.
        Skips chunks with IDs already in the collection.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match")

        collection = self._get_collection()
        existing_ids = set(collection.get(include=[])["ids"])

        new_chunks = []
        new_embeddings = []
        for chunk, emb in zip(chunks, embeddings):
            cid = chunk["chunk_id"]
            if cid not in existing_ids:
                new_chunks.append(chunk)
                new_embeddings.append(emb)

        if not new_chunks:
            logger.debug("All chunks already exist; nothing added")
            return 0

        # Chroma metadata values must be str/int/float/bool
        documents = [c["text"] for c in new_chunks]
        ids = [c["chunk_id"] for c in new_chunks]
        metadatas = []
        for c in new_chunks:
            meta = {
                "doc_id": str(c.get("doc_id", "")),
                "source": str(c.get("source", "")),
                "page_number": int(c.get("page_number", 1)),
                "chunk_index": int(c.get("chunk_index", 0)),
                "file_type": str(c.get("file_type", "")),
            }
            metadatas.append(meta)

        collection.add(
            documents=documents,
            embeddings=new_embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(new_chunks)} chunks to collection '{self.collection_name}'")
        return len(new_chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search.
        Returns list of result dicts with text, metadata, and distance.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            logger.warning("Vector store is empty; no results")
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        return [
            {
                "chunk_id": results["ids"][0][i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
                "score": max(0.0, 1.0 - dists[i]),  # cosine distance → similarity
                "retrieval_method": "dense",
            }
            for i in range(len(docs))
        ]

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document."""
        collection = self._get_collection()
        try:
            existing = collection.get(where={"doc_id": doc_id}, include=[])
            ids = existing["ids"]
            if ids:
                collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks for doc_id={doc_id}")
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting doc_id={doc_id}: {e}")
            return 0

    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique documents currently indexed."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", ""),
                    "file_type": meta.get("file_type", ""),
                }
        return list(seen.values())

    def count(self) -> int:
        return self._get_collection().count()

    def reset(self) -> None:
        """Delete all documents. Use with caution."""
        collection = self._get_collection()
        collection.delete(where={"doc_id": {"$ne": ""}})
        logger.warning("Vector store reset: all chunks deleted")
