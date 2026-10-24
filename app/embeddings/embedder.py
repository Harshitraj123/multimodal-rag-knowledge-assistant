"""
Text embedding using sentence-transformers.
Configurable model, consistent query/document embedding,
lazy loading to avoid startup delays.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a cached SentenceTransformer instance."""
    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        t = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            _model_cache[model_name] = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded in {time.time() - t:.2f}s")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e
    return _model_cache[model_name]


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Embed a list of texts.
    Returns list of float vectors.
    """
    if not texts:
        return []
    model = get_embedder(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[float]:
    """
    Embed a single query string.
    Uses identical model config as document embedding for consistency.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty")
    results = embed_texts([query], model_name=model_name)
    return results[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
