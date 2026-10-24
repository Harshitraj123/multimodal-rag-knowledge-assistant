# Learning Guide

This guide walks through the project systematically so you can understand
every component before writing your own implementation.

## Step 1: Understand the Project Structure

Start by reading:
1. `README.md` — overview and commands
2. `docs/ARCHITECTURE.md` — pipeline diagrams
3. `docs/DESIGN_DECISIONS.md` — why things were built this way

Then explore the directory structure to see how modules relate.

## Step 2: Configuration (app/config/)

**Files:** `settings.py`, `logging_config.py`

Key concepts:
- `pydantic-settings` loads from environment variables and `.env`
- `Settings.report_capabilities()` returns a sanitized status dict (no secrets)
- `settings.llm_available` and `settings.vision_available` control degraded modes

**Experiment:** What happens if you set `GROQ_API_KEY=""` and call `settings.llm_available`?

## Step 3: Document Loading (app/ingestion/loaders.py)

Key concepts:
- `RawDocument` dataclass holds pages, tables, images, and extraction errors
- Each file type has a dedicated loader function
- pdfplumber extracts text and tables AND renders pages as images
- Extraction errors are collected (not raised) to allow partial success

**Experiment:** Call `load_document()` on a PDF and inspect the `pages` and `tables` fields.

## Step 4: Text Chunking (app/chunking/splitter.py)

Key concepts:
- Paragraph-first splitting preserves semantic boundaries
- Sentence-boundary splitting handles oversized paragraphs
- Overlap creates context continuity across chunk boundaries
- `chunk_table()` preserves headers in every chunk

**Why this matters:** Chunk size affects retrieval quality more than almost
any other parameter. Too small = lost context. Too large = noisy retrieval.

**Experiment:** Change `chunk_size` and observe how many chunks a document produces.

## Step 5: Embeddings (app/embeddings/embedder.py)

Key concepts:
- SentenceTransformer encodes text to dense float vectors
- `normalize_embeddings=True` makes cosine similarity well-behaved
- Query and document embeddings use the same model for consistency
- The model is cached (loaded once per process)

**Why normalization matters:** Without L2 normalization, dot product ≠ cosine similarity.

## Step 6: Vector Storage (app/storage/vector_store.py)

Key concepts:
- ChromaDB stores documents, embeddings, and metadata together
- Cosine distance is configured at collection creation (not query time)
- `score = 1.0 - distance` converts Chroma's distance to a similarity
- IDs must be stable to support duplicate detection and deletion

**Inspect:** After ingestion, open `data/chroma_db/` and see the SQLite files Chroma creates.

## Step 7: BM25 Retrieval (app/retrieval/sparse.py)

Key concepts:
- BM25 scores documents by term frequency and inverse document frequency
- `tokenize()` lowercases and splits on word boundaries
- The index is rebuilt from a persisted manifest on startup
- BM25 excels at exact matches, named entities, and technical terms

**When does BM25 beat dense retrieval?**
- Exact technical acronyms (e.g., "RFC 4271", "CUDA", "BM25")
- Named entities that embeddings generalize over
- Rare terminology not seen during embedding training

## Step 8: Hybrid Retrieval & RRF (app/retrieval/hybrid.py, fusion.py)

Key concepts:
- Dense and BM25 results are passed separately to RRF
- RRF formula: score(d) = Σ 1/(k + rank_r(d)) across all result lists
- k=60 prevents very high-ranked items from completely dominating
- RRF does NOT require score normalization (ranks are always integers)

**Test it:** Run the same query through dense-only and hybrid in the Retrieval Explorer.
Observe which documents each strategy surfaces.

## Step 9: MMR (app/retrieval/mmr.py)

Key concepts:
- MMR selects next result by maximizing relevance minus redundancy
- lambda=1.0 → pure top-k by relevance (no diversification)
- lambda=0.0 → pure diversity (ignores query relevance entirely)
- lambda=0.5 → balanced (recommended default)

**When to use MMR:** When you have many topically similar documents and want
to cover more perspectives rather than returning near-duplicates.

## Step 10: Reranking (app/retrieval/reranker.py)

Key concepts:
- Stage 1 (retrieval): returns broad candidates quickly
- Stage 2 (reranking): cross-encoder jointly encodes query+passage
- Cross-encoders are more accurate than bi-encoders (no semantic gap)
- Cross-encoders are O(n) per query — latency scales with candidate count

**The two-stage tradeoff:**
- Bi-encoder (retrieval): ~45ms, ~85% accuracy
- Cross-encoder reranking on top: ~285ms, ~93% accuracy

## Step 11: Query Processing (app/query_processing/)

Each strategy is independently testable:

| Strategy | File | When to use |
|----------|------|-------------|
| rewrite | rewrite.py | Ambiguous or poorly-phrased queries |
| expand | expansion.py | Short queries with narrow terminology |
| multi_query | multi_query.py | Queries that could be phrased many ways |
| decompose | decomposition.py | Complex multi-part questions |
| hyde | hyde.py | Queries with no obvious keywords |

**HyDE risk:** The hypothetical document may hallucinate content.
This is why HyDE results use the hypothetical text for retrieval only —
the REAL retrieved documents are used for the final answer.

## Step 12: Generation & Citations (app/generation/)

Key concepts:
- `AnswerGenerator` formats retrieved chunks as numbered evidence [1], [2], ...
- The model is instructed to cite only the provided evidence
- `validate_citations()` checks that cited numbers correspond to real sources
- Invalid citations are marked `[N:INVALID]`, not silently removed

**What makes this better than a naive chatbot:**
- The model cannot invent document names or page numbers
- Invalid citations are detected programmatically
- When evidence is insufficient, the system says so explicitly

## Step 13: Evaluation (app/evaluation/)

Key concepts:
- All retrieval metrics (Hit@k, Recall@k, MRR, nDCG@k) are deterministic
- `faithfulness_heuristic` is a word-overlap proxy — not equivalent to LLM judge
- The evaluation dataset is version-controlled JSON
- Results are saved to JSON for reproducibility

**Experiment:** Add 5 new evaluation cases, run the benchmark, and compare configs.

## Step 14: API (app/api/)

Key concepts:
- All component initialization happens in `dependencies.py`
- Singletons ensure one embedding model per process
- `DocumentService` wraps pipeline operations for the API layer
- Pydantic validates all input before reaching business logic

## Step 15: Frontend (frontend/streamlit_app.py)

Key concepts:
- `@st.cache_resource` initializes the backend once and caches it
- Session state tracks conversation history across chat turns
- The Retrieval Explorer runs real retrieval (not mock data)
- The Evaluation Lab shows actual benchmark results

## Next Steps for Your Own Implementation

Once you understand this codebase:

1. Replace ChromaDB with FAISS to understand the difference
2. Try a different embedding model (e.g., `all-mpnet-base-v2`) and measure the impact
3. Implement a proper RAGAS evaluation using RAGAS library
4. Add streaming response support to the API
5. Implement a proper LLM-as-judge faithfulness evaluator
