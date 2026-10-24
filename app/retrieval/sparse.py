"""
BM25 sparse retrieval.
Good for exact terminology, named entities, and technical terms.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """
    BM25 keyword-based retrieval.
    Can be persisted and reconstructed from a manifest.
    """

    def __init__(self):
        self._bm25 = None
        self._corpus: list[str] = []
        self._ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    def build(
        self,
        corpus: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Build BM25 index from corpus."""
        if not corpus:
            logger.warning("Empty corpus; BM25 index not built")
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")

        tokenized = [tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus = corpus
        self._ids = ids
        self._metadatas = metadatas
        logger.info(f"BM25 index built with {len(corpus)} documents")

    def add_documents(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents and rebuild the index."""
        existing_ids = set(self._ids)
        new_texts, new_ids, new_metas = [], [], []
        for t, i, m in zip(texts, ids, metadatas):
            if i not in existing_ids:
                new_texts.append(t)
                new_ids.append(i)
                new_metas.append(m)
        if not new_texts:
            return
        self._corpus.extend(new_texts)
        self._ids.extend(new_ids)
        self._metadatas.extend(new_metas)
        self.build(self._corpus, self._ids, self._metadatas)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top_k documents by BM25 score."""
        if self._bm25 is None or not self._corpus:
            logger.warning("BM25 index is empty")
            return []
        t = time.time()
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        elapsed = time.time() - t
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "chunk_id": self._ids[idx],
                "text": self._corpus[idx],
                "metadata": self._metadatas[idx],
                "bm25_score": float(scores[idx]),
                "score": float(scores[idx]),
                "retrieval_method": "sparse",
                "retrieval_latency_s": elapsed,
            })
        logger.info(f"BM25 retrieval: {len(results)} results in {elapsed:.3f}s")
        return results

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)
