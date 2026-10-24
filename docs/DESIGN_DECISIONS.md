# Design Decisions

## 1. Why Reciprocal Rank Fusion instead of score interpolation?

Dense and sparse retrieval produce scores on incomparable scales:
- BM25 scores are unbounded positive numbers
- Cosine similarity is bounded to [0, 1] after normalization

Score interpolation (e.g., 0.7 * dense_score + 0.3 * sparse_score) would require
normalizing across both distributions, which is brittle and query-dependent.

RRF uses only rank positions, making it robust to distribution differences.
The k=60 constant is from the original RRF paper (Cormack et al., 2009) and
has proven stable across many retrieval settings.

**Alternative considered:** Learned linear interpolation via a validation set.
**Rejected because:** Requires labeled training data; overkill for a portfolio system.

## 2. Why ChromaDB instead of FAISS or Pinecone?

ChromaDB provides persistent local storage out of the box with a simple Python API.
FAISS requires manual index serialization and does not store metadata.
Pinecone requires external infrastructure and billing.

For a local, portfolio-grade system ChromaDB is the appropriate choice.

**Limitation acknowledged:** ChromaDB has lower throughput than FAISS for large-scale
production workloads. The storage layer is deliberately isolated to allow substitution.

## 3. Why keep text and visual indexes separate?

Text/table chunks and visual assets have different retrieval characteristics:
- Text queries should search text chunks primarily
- Visual queries need image-specific metadata (image bytes, page position)
- Mixing them in one collection would require awkward filtering

Using two ChromaDB collections keeps retrieval logic clean and understandable.

## 4. Why sentence-transformers/all-MiniLM-L6-v2 as default?

- 384-dimensional vectors (fast)
- Runs locally without API keys
- Well-tested on semantic similarity tasks
- Available via sentence-transformers without additional registration

**Limitation:** 256-token max sequence; very long passages are truncated.

## 5. Why bounded conversation history?

Sending unlimited conversation history to the LLM:
- Increases cost and latency quadratically
- Can cause context window overflow
- Can cause the model to drift toward earlier topics

The system limits history to the last 20 messages (10 turns).
This is configurable via the `reformulate_for_history` method.

## 6. Why not LangChain as the primary framework?

LangChain is powerful but adds indirection that makes debugging harder.
For a portfolio project, directly implementing the pipeline makes the
architecture transparent and easier to explain in interviews.

LangChain is used nowhere in this implementation. All retrieval, fusion,
query processing, and generation are implemented directly.

## 7. Why pdfplumber instead of PyMuPDF or pypdf?

pdfplumber provides high-quality table extraction and page rendering in one package.
PyMuPDF is faster but less accessible for table-specific extraction.
pypdf is text-only and doesn't render page images.

## 8. Why separate LLM and vision providers?

Text-generation models (Groq/Llama) may not support image inputs.
Vision-capable models (GPT-4o-mini) may be more expensive per call.

Keeping them separate allows:
- Text generation without vision API costs
- Visual analysis with a different provider or model
- Independent configuration and fallback handling
