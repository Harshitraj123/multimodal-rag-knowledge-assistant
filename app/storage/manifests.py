"""
BM25 index persistence.
Saves tokenized documents and reconstructs the sparse index on restart.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_bm25_manifest(
    manifest_path: str | Path,
    corpus: list[str],
    ids: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Save BM25 corpus data to disk."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "corpus": corpus,
        "ids": ids,
        "metadatas": metadatas,
    }
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.debug(f"BM25 manifest saved: {len(corpus)} documents")


def load_bm25_manifest(
    manifest_path: str | Path,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Load BM25 corpus data from disk. Returns (corpus, ids, metadatas)."""
    path = Path(manifest_path)
    if not path.exists():
        return [], [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["corpus"], data["ids"], data["metadatas"]
    except Exception as e:
        logger.error(f"Failed to load BM25 manifest: {e}")
        return [], [], []
