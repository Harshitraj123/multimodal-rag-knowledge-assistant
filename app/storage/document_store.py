"""
Persistent document registry.
Stores document metadata and manages duplicate detection.
Uses JSON file for simplicity and portability.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Tracks ingested documents and their metadata.
    Persists to a JSON file alongside the vector store.
    """

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self._docs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                self._docs = json.loads(self.store_path.read_text(encoding="utf-8"))
                logger.debug(f"Loaded {len(self._docs)} documents from registry")
            except Exception as e:
                logger.error(f"Failed to load document store: {e}")
                self._docs = {}

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(self._docs, indent=2, default=str),
            encoding="utf-8",
        )

    def register(self, doc_id: str, metadata: dict[str, Any]) -> bool:
        """
        Register a document. Returns True if new, False if duplicate.
        Does not update existing entries to prevent accidental overwrite.
        """
        if doc_id in self._docs:
            logger.info(f"Document {doc_id} already registered; skipping")
            return False
        self._docs[doc_id] = {
            **metadata,
            "doc_id": doc_id,
            "ingested_at": time.time(),
        }
        self._save()
        return True

    def get(self, doc_id: str) -> Optional[dict[str, Any]]:
        return self._docs.get(doc_id)

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._docs.values())

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._save()
            logger.info(f"Removed document {doc_id} from registry")
            return True
        return False

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def count(self) -> int:
        return len(self._docs)
