# [SYNTHETIC DATA - NOT REAL] RAG System Architecture Overview

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
