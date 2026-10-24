# Limitations

This document describes concrete limitations of the current implementation.

## 1. Vision Analysis Requires External API

Visual reasoning (answering questions about charts, diagrams, and images)
requires a configured vision-capable API key (default: OpenAI GPT-4o-mini).

Without a vision API key:
- Documents with images are still ingested
- Page images are stored but cannot be analyzed
- Questions requiring visual analysis will receive text-only answers
- The system correctly discloses this limitation

## 2. BM25 Index Rebuild on Document Deletion

The BM25 index is an in-memory structure that is persisted as a manifest JSON file.
When a document is deleted, the BM25 index is marked stale. It does not immediately
rebuild. The index will be missing the deleted document's chunks until the next
application restart with a fresh ingest.

**Workaround:** Re-run ingestion after deletions, or restart the application.

## 3. Context Window and Top-K Limits

Only the top-k retrieved chunks are passed to the LLM per query.
For very long documents, relevant information in lower-ranked chunks may be missed.

Increasing `TOP_K` improves coverage but increases LLM cost and latency.
Very high top-k values may exceed the LLM's context window.

## 4. Evaluation Dataset Is Small and Synthetic

The provided evaluation dataset has 10 synthetic cases.
Results from this dataset should not be extrapolated to production performance.

Real-world benchmarking requires:
- A larger, domain-representative corpus
- Human-curated relevance labels
- Multiple annotators for agreement

## 5. Prompt Injection Risk

Document content is treated as data, not instructions.
The system uses a fixed system prompt instructing the model to answer from evidence only.
However, a sophisticated adversarial document could potentially manipulate the LLM.

This risk cannot be fully eliminated with prompt-based mitigations.
Do not use this system with adversarial or untrusted documents in a production context
without additional security measures.

## 6. Single-Node Local Deployment

The system is designed for local, single-user deployment.
It does not support:
- Multi-user concurrent access
- Horizontal scaling
- Authentication or authorization
- Rate limiting
- Production security hardening

## 7. Model Availability

LLM and vision provider models may be deprecated or renamed by their providers.
Check official provider documentation for current model names.

## 8. Large PDF Performance

Very large PDFs (100+ pages) with page-level image rendering can be slow to ingest
and produce large amounts of stored image data.

## 9. Streaming Responses Not Supported

The API and frontend do not support streaming. The full answer is generated before
being returned. For very long answers, this can cause noticeable latency.

## 10. Evaluation Metrics Are Approximate

- `faithfulness_heuristic` is a word-overlap proxy, not a true faithfulness metric.
- True faithfulness evaluation requires an LLM judge and carries its own costs.
- Citation precision is deterministic but only validates citation number existence,
  not the semantic accuracy of the citation.
