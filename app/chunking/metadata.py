"""Metadata attachment and normalization for chunks."""
from __future__ import annotations
from typing import Any


def attach_metadata(chunk: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge extra metadata into a chunk dict."""
    return {**chunk, **extra}


def normalize_source_id(source: str, page: int, chunk_index: int) -> str:
    """Create a stable source identifier."""
    safe = source.replace("/", "_").replace("\\", "_")
    return f"{safe}::p{page}::c{chunk_index}"
