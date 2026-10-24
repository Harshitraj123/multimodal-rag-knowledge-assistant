"""
Text chunking with configurable size and overlap.
Uses sentence-aware splitting to avoid breaking sentences mid-way.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: str = "\n\n",
) -> list[str]:
    """
    Split text into overlapping chunks.
    Prefers splitting on paragraph breaks, then sentence boundaries.
    Returns list of non-empty chunk strings.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []

    # Split on paragraph boundaries first
    paragraphs = text.split(separator)
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)

        if current_len + para_len + 2 <= chunk_size:
            current.append(para)
            current_len += para_len + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            # Para is larger than chunk_size: split by sentences
            if para_len > chunk_size:
                sentence_chunks = _split_by_sentences(para, chunk_size, chunk_overlap)
                chunks.extend(sentence_chunks)
                current = []
                current_len = 0
            else:
                current = [para]
                current_len = para_len

    if current:
        chunks.append("\n\n".join(current))

    # Apply overlap: prepend tail of previous chunk to current
    chunks = _apply_overlap(chunks, chunk_overlap)

    return [c for c in chunks if c.strip()]


def _split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split oversized text by sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > chunk_size:
                # Hard split by characters
                for i in range(0, len(sentence), chunk_size - chunk_overlap):
                    chunks.append(sentence[i : i + chunk_size])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend tail of previous chunk to current for context continuity."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
        combined = prev_tail + " " + chunks[i]
        result.append(combined)
    return result


def chunk_table(table: dict, max_rows_per_chunk: int = 20) -> list[str]:
    """
    Convert a table dict into text chunks.
    Preserves header + up to max_rows_per_chunk rows per chunk.
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    source = table.get("source", "unknown")
    page = table.get("page_number", 1)

    if not headers and not rows:
        return []

    header_str = " | ".join(str(h) for h in headers) if headers else ""
    separator = "-" * max(len(header_str), 10)

    chunks = []
    for i in range(0, max(len(rows), 1), max_rows_per_chunk):
        batch = rows[i : i + max_rows_per_chunk]
        lines = [f"Table from {source} (page {page}):"]
        if header_str:
            lines.append(header_str)
            lines.append(separator)
        for row in batch:
            lines.append(" | ".join(str(c) for c in row))
        chunks.append("\n".join(lines))

    return chunks
