"""
Creates a synthetic demo corpus for testing the RAG system.
All content is explicitly labelled SYNTHETIC.
Does not represent any real company or actual data.

Generates:
- A multi-page technical report PDF (synthetic_tech_report.pdf)
- A dataset CSV (synthetic_performance_data.csv)
- A markdown overview (synthetic_overview.md)
"""
from __future__ import annotations

import csv
import io
import sys
import os
from pathlib import Path

# Ensure the project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "data" / "demo_corpus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_LABEL = "[SYNTHETIC DATA - NOT REAL]"


def create_markdown_overview():
    """Create a markdown document about the synthetic RAG system."""
    content = f"""# {SYNTHETIC_LABEL} RAG System Architecture Overview

## Introduction

This is a **synthetic** technical overview created for demonstration purposes.
It describes a fictional document intelligence system to test retrieval capabilities.

## System Architecture

The fictional RAG system consists of the following components:

1. **Document Ingestion Layer**: Processes PDF, DOCX, CSV, and image files
2. **Embedding Engine**: Uses sentence-transformers for dense vector encoding
3. **Vector Store**: Persistent ChromaDB instance with cosine similarity
4. **Sparse Retriever**: BM25-based keyword matching
5. **Hybrid Fusion**: Reciprocal Rank Fusion combining dense and sparse
6. **Answer Generator**: Groq-powered grounded generation with citations

## Architecture Diagram Description

The architecture follows a pipeline pattern:
- Input documents flow through the ingestion layer
- Text chunks are embedded and stored in the vector database
- Queries trigger parallel dense and sparse retrieval
- Results are fused, optionally reranked, then passed to the LLM
- The LLM generates a grounded answer with numbered citations

## Performance Characteristics

Based on synthetic benchmarks on the demo corpus:

| Configuration | MRR  | Hit@5 | Recall@5 |
|---------------|------|-------|----------|
| Dense Only    | 0.72 | 0.85  | 0.71     |
| Sparse Only   | 0.65 | 0.78  | 0.63     |
| Hybrid RRF    | 0.81 | 0.92  | 0.83     |
| Hybrid + MMR  | 0.79 | 0.90  | 0.85     |

*NOTE: These are synthetic benchmark values for demonstration only.*

## Key Design Decisions

### Why Hybrid Retrieval?
Dense retrieval handles semantic similarity well but struggles with exact terms.
BM25 handles exact matches but misses semantic equivalents.
Hybrid combines both strengths via Reciprocal Rank Fusion.

### Why Reciprocal Rank Fusion?
RRF does not require score normalization, which is critical when combining
BM25 scores (unbounded) with cosine similarity scores (0 to 1).

### Why ChromaDB?
ChromaDB provides persistent local storage, simple Python API, and supports
metadata filtering without external infrastructure.

## Limitations

- Single-node deployment only
- No streaming responses
- Vision analysis requires external API credentials
- Reranking adds latency (200-400ms per query)

## Contact

This is synthetic content. For the real implementation, see the project README.
"""
    path = OUTPUT_DIR / "synthetic_overview.md"
    path.write_text(content, encoding="utf-8")
    print(f"Created: {path}")


def create_csv_data():
    """Create synthetic performance benchmark CSV."""
    rows = [
        ["config", "mrr", "hit_at_5", "recall_at_5", "ndcg_at_5", "avg_latency_ms"],
        ["dense_only", 0.720, 0.850, 0.710, 0.769, 45],
        ["sparse_only", 0.650, 0.780, 0.630, 0.698, 12],
        ["hybrid_rrf", 0.810, 0.920, 0.830, 0.867, 58],
        ["hybrid_mmr", 0.790, 0.900, 0.850, 0.851, 72],
        ["hybrid_rerank", 0.840, 0.930, 0.850, 0.893, 285],
        ["rewrite+hybrid", 0.820, 0.925, 0.840, 0.879, 320],
        ["hyde+hybrid", 0.825, 0.928, 0.845, 0.882, 450],
    ]
    path = OUTPUT_DIR / "synthetic_performance_data.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Created: {path}")


def create_text_report():
    """Create a synthetic multi-page text report."""
    content = f"""{SYNTHETIC_LABEL}
SYNTHETIC TECHNICAL REPORT: Document Intelligence System v2.0
==============================================================

PAGE 1: EXECUTIVE SUMMARY
--------------------------

This synthetic report describes a fictional document intelligence platform
designed to demonstrate Retrieval-Augmented Generation (RAG) capabilities.

The system supports the following document types:
- PDF documents (text, tables, rendered page images)
- Word documents (DOCX) with tables and embedded images
- CSV data files with structured tabular data
- Markdown and plain text files
- PowerPoint presentations (PPTX)
- Standalone image files (PNG, JPG, JPEG)

Key capabilities demonstrated:
1. Hybrid dense/sparse retrieval
2. Visual content understanding
3. Table extraction and retrieval
4. Source citation with validation
5. Multi-turn conversation support

PAGE 2: TECHNICAL SPECIFICATIONS
----------------------------------

EMBEDDING MODEL SPECIFICATIONS:
- Model: sentence-transformers/all-MiniLM-L6-v2
- Embedding Dimension: 384
- Max Sequence Length: 256 tokens
- Similarity Metric: Cosine similarity
- Normalization: L2-normalized vectors

CHUNKING PARAMETERS:
- Default chunk size: 512 characters
- Overlap: 64 characters
- Strategy: Paragraph-first, then sentence-boundary splitting
- Table chunking: 20 rows per chunk with header repetition

BM25 PARAMETERS:
- Implementation: BM25Okapi
- Tokenization: Whitespace + punctuation, lowercased
- k1: 1.5 (BM25Okapi default)
- b: 0.75 (BM25Okapi default)

RRF PARAMETERS:
- k constant: 60 (standard from original RRF paper)
- Input lists: dense + sparse ranked results
- Output: score = sum(1 / (60 + rank))

PAGE 3: RETRIEVAL BENCHMARKS
------------------------------

SYNTHETIC EVALUATION DATASET:
- 20 evaluation questions
- 5 difficulty levels: easy (6), medium (8), hard (6)
- Question types: text (10), table (5), image (3), cross-modal (2)

RETRIEVAL CONFIGURATION COMPARISON:

Configuration         | MRR   | Hit@5 | Recall@5 | Latency
----------------------|-------|-------|----------|--------
Dense Only            | 0.720 | 0.850 | 0.710    | 45ms
Sparse (BM25)         | 0.650 | 0.780 | 0.630    | 12ms
Hybrid RRF            | 0.810 | 0.920 | 0.830    | 58ms
Hybrid + MMR          | 0.790 | 0.900 | 0.850    | 72ms
Hybrid + Reranking    | 0.840 | 0.930 | 0.850    | 285ms
Query Rewriting       | 0.820 | 0.925 | 0.840    | 320ms
HyDE                  | 0.825 | 0.928 | 0.845    | 450ms

NOTE: All values above are SYNTHETIC and for demonstration only.
Real benchmarks require actual API credentials and a real document corpus.

PAGE 4: VISUAL CONTENT HANDLING
---------------------------------

The system supports three types of visual content:

TYPE 1: PAGE RENDERS
PDF pages are rendered at 150 DPI using pdfplumber.
Each rendered page is indexed with a vision-model description.
Page renders capture content that text extraction misses:
- Mathematical formulas
- Complex layouts
- Handwritten annotations
- Diagrams embedded in text flow

TYPE 2: EMBEDDED IMAGES
DOCX and PPTX files may contain embedded images.
These are extracted and indexed separately with their page/slide context.

TYPE 3: STANDALONE IMAGES
PNG, JPG, and JPEG files are treated as visual documents.
The entire file becomes one visual asset with auto-generated description.

VISUAL RETRIEVAL PROCESS:
1. Query is embedded and compared to visual description embeddings
2. Top-k visual assets are retrieved
3. Original image bytes are decoded from ChromaDB metadata
4. Vision model receives original image + user question
5. Response is grounded in actual visual evidence

PAGE 5: LIMITATIONS AND KNOWN ISSUES
--------------------------------------

1. VISION API DEPENDENCY
   Visual reasoning requires an external vision-capable model API.
   Without credentials, visual content descriptions are stored but
   questions cannot be answered with visual evidence.

2. BM25 REBUILD ON DELETE
   Deleting a document marks it as deleted in the registry and removes
   its vector store chunks. The BM25 index is rebuilt on next ingest.
   For very large corpora, this may be slow.

3. CONTEXT WINDOW LIMITS
   Very long documents may produce many chunks.
   Only top-k chunks are sent to the LLM per query.
   Information in lower-ranked chunks may be missed.

4. EVALUATION LIMITATIONS
   The provided evaluation dataset is synthetic and small (20 cases).
   Real-world performance depends heavily on document quality and corpus size.

5. PROMPT INJECTION RISK
   Document content is treated as untrusted data.
   A fixed system prompt instructs the model to answer only from evidence.
   However, sophisticated prompt injection in documents cannot be fully prevented.

END OF SYNTHETIC REPORT
"""
    path = OUTPUT_DIR / "synthetic_tech_report.txt"
    path.write_text(content, encoding="utf-8")
    print(f"Created: {path}")


def create_evaluation_dataset():
    """Create evaluation questions for the demo corpus."""
    import json
    cases = [
        {
            "case_id": "eval_001",
            "question": "What embedding model is used in the synthetic system?",
            "expected_sources": ["synthetic_tech_report.txt", "synthetic_overview.md"],
            "expected_pages": [1],
            "required_evidence": "all-MiniLM-L6-v2",
            "answer_type": "text",
            "difficulty": "easy",
            "reference_answer": "The system uses sentence-transformers/all-MiniLM-L6-v2 with 384-dimensional embeddings.",
            "notes": "Direct lookup from technical specs"
        },
        {
            "case_id": "eval_002",
            "question": "What is the RRF k constant and why is it used?",
            "expected_sources": ["synthetic_tech_report.txt"],
            "expected_pages": [3],
            "required_evidence": "60",
            "answer_type": "text",
            "difficulty": "medium",
            "reference_answer": "The RRF k constant is 60, from the original RRF paper. It prevents very high-ranked items from dominating the score.",
            "notes": "Requires understanding of RRF"
        },
        {
            "case_id": "eval_003",
            "question": "Which retrieval configuration had the highest MRR in the benchmark table?",
            "expected_sources": ["synthetic_tech_report.txt", "synthetic_performance_data.csv"],
            "expected_pages": [3],
            "required_evidence": "Hybrid + Reranking, 0.840",
            "answer_type": "table",
            "difficulty": "easy",
            "reference_answer": "Hybrid + Reranking had the highest MRR at 0.840.",
            "notes": "Tests table retrieval"
        },
        {
            "case_id": "eval_004",
            "question": "What are the five limitations described in the system report?",
            "expected_sources": ["synthetic_tech_report.txt"],
            "expected_pages": [5],
            "required_evidence": "VISION API DEPENDENCY, BM25 REBUILD, CONTEXT WINDOW, EVALUATION LIMITATIONS, PROMPT INJECTION",
            "answer_type": "text",
            "difficulty": "medium",
            "reference_answer": "The five limitations are: vision API dependency, BM25 rebuild on delete, context window limits, evaluation limitations, and prompt injection risk.",
            "notes": "Multi-item retrieval"
        },
        {
            "case_id": "eval_005",
            "question": "What is the average latency of dense-only retrieval?",
            "expected_sources": ["synthetic_performance_data.csv", "synthetic_tech_report.txt"],
            "expected_pages": [1],
            "required_evidence": "45",
            "answer_type": "table",
            "difficulty": "easy",
            "reference_answer": "Dense-only retrieval has an average latency of 45ms.",
            "notes": "Exact numeric lookup from CSV"
        },
        {
            "case_id": "eval_006",
            "question": "Why does the system use hybrid retrieval instead of dense-only?",
            "expected_sources": ["synthetic_overview.md"],
            "expected_pages": [1],
            "required_evidence": "exact matches, semantic equivalents, Reciprocal Rank Fusion",
            "answer_type": "text",
            "difficulty": "medium",
            "reference_answer": "Hybrid retrieval combines dense (semantic similarity) and sparse (exact matching) approaches. Dense handles semantic equivalents; BM25 handles exact terms. Both strengths are combined via RRF.",
            "notes": "Requires synthesis from design decisions section"
        },
        {
            "case_id": "eval_007",
            "question": "What is the embedding dimension of the all-MiniLM-L6-v2 model?",
            "expected_sources": ["synthetic_tech_report.txt"],
            "expected_pages": [2],
            "required_evidence": "384",
            "answer_type": "text",
            "difficulty": "easy",
            "reference_answer": "The embedding dimension is 384.",
            "notes": "Direct lookup"
        },
        {
            "case_id": "eval_008",
            "question": "What document types does the system support?",
            "expected_sources": ["synthetic_tech_report.txt", "synthetic_overview.md"],
            "expected_pages": [1],
            "required_evidence": "PDF, DOCX, CSV, Markdown, PPTX, PNG, JPG",
            "answer_type": "text",
            "difficulty": "easy",
            "reference_answer": "The system supports PDF, DOCX, CSV, Markdown (.md), plain text (.txt), PPTX, PNG, JPG, and JPEG.",
            "notes": "List enumeration"
        },
        {
            "case_id": "eval_009",
            "question": "What are the three types of visual content handled by the system?",
            "expected_sources": ["synthetic_tech_report.txt"],
            "expected_pages": [4],
            "required_evidence": "PAGE RENDERS, EMBEDDED IMAGES, STANDALONE IMAGES",
            "answer_type": "text",
            "difficulty": "medium",
            "reference_answer": "The three types are: page renders (PDF pages at 150 DPI), embedded images (from DOCX/PPTX), and standalone image files (PNG/JPG).",
            "notes": "Requires reading page 4"
        },
        {
            "case_id": "eval_010",
            "question": "What is the HyDE retrieval latency compared to dense-only?",
            "expected_sources": ["synthetic_performance_data.csv", "synthetic_tech_report.txt"],
            "expected_pages": [3],
            "required_evidence": "450, 45",
            "answer_type": "table",
            "difficulty": "hard",
            "reference_answer": "HyDE has a latency of 450ms versus 45ms for dense-only — approximately 10x slower due to the hypothesis generation step.",
            "notes": "Requires comparison between two table rows"
        },
    ]
    path = PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Created evaluation dataset: {path} ({len(cases)} cases)")


if __name__ == "__main__":
    print("Creating synthetic demo corpus...")
    create_markdown_overview()
    create_csv_data()
    create_text_report()
    create_evaluation_dataset()
    print(f"\nDemo corpus created in: {OUTPUT_DIR}")
    print("All content is synthetic and clearly labelled.")
