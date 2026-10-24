# Architecture

## System Overview

The Multimodal RAG Knowledge Assistant is a local-first document intelligence system
built on a modular pipeline architecture. It supports ingestion of multiple document types,
hybrid retrieval, visual reasoning, grounded answer generation, and measurable evaluation.

## Pipeline Diagrams

### Ingestion Pipeline

```
File Input
    │
    ▼
Document Loader (app/ingestion/loaders.py)
    │  PDF → pdfplumber (text + tables + page images)
    │  DOCX → python-docx (paragraphs + tables + embedded images)
    │  CSV → csv.DictReader (structured rows)
    │  TXT/MD → direct text read
    │  PPTX → python-pptx (slide text + tables)
    │  PNG/JPG → raw bytes
    ▼
ProcessedDocument (app/ingestion/document_processor.py)
    │
    ├─ Text chunks ──→ Embedder ──→ ChromaDB (dense index)
    │                           └──→ BM25Retriever (sparse index)
    │
    ├─ Table chunks ─→ Embedder ──→ ChromaDB
    │                          └──→ BM25
    │
    └─ Images ──────→ VisionProvider (description) ──→ VisualRetriever (ChromaDB)
```

### Query Pipeline

```
User Query
    │
    ▼
Conversation History Reformulation (optional)
    │
    ▼
Query Strategy (rewrite | expand | multi_query | decompose | hyde)
    │
    ▼
Retrieval
    ├─ Dense: embed(query) → ChromaDB cosine search
    ├─ Sparse: BM25Okapi.get_scores(tokens)
    └─ Hybrid: Dense + Sparse → Reciprocal Rank Fusion
    │
    ▼
Optional Post-Processing
    ├─ MMR: diversity-aware re-selection
    └─ Reranking: CrossEncoder second-stage scoring
    │
    ▼
Visual Retrieval (parallel)
    │
    ▼
Answer Generator
    ├─ LLM not configured → retrieval-only mode
    └─ LLM available → RAG_ANSWER_PROMPT + citation validation
    │
    ▼
GeneratedAnswer (citations validated, sources attached)
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `app/config/` | Settings, logging configuration |
| `app/ingestion/` | File loading, document processing |
| `app/chunking/` | Text splitting, table-to-text, overlap |
| `app/embeddings/` | SentenceTransformer embedding |
| `app/storage/` | ChromaDB, document registry, BM25 manifest |
| `app/retrieval/` | Dense, sparse, hybrid, MMR, reranker, visual |
| `app/query_processing/` | Rewrite, expand, multi-query, decompose, HyDE |
| `app/generation/` | LLM/vision providers, prompts, citations, answer generator |
| `app/pipeline/` | Ingestion and RAG orchestration |
| `app/evaluation/` | Metrics, dataset, benchmark runner |
| `app/api/` | FastAPI routes, schemas, dependency injection |
| `app/services/` | Document management service |
| `frontend/` | Streamlit UI |

## Key Design Decisions

See `docs/DESIGN_DECISIONS.md` for detailed rationale.
