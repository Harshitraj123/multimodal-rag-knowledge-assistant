# Interview Preparation

These questions and answers are grounded in this project's actual implementation.

## Q: Why use RAG instead of passing the entire document to an LLM?

**Context window limits:** Most LLMs support 8k-128k tokens. A large PDF may have
millions of characters. Retrieval selects the most relevant sections.

**Cost:** Sending full documents on every query is expensive. Retrieval reduces tokens sent.

**Accuracy:** LLMs can lose information from the middle of long contexts ("lost in the middle"
problem). Retrieved top-k chunks keep relevant content near the prompt's attention focus.

**Updatability:** New documents can be ingested without retraining the model.

## Q: Why is chunking important?

Chunking affects both retrieval precision and context quality:

**Too small (< 100 chars):** Chunks lack context. Embeddings represent fragments, not ideas.
Retrieval returns disconnected phrases.

**Too large (> 1000 chars):** One chunk may contain many topics. Retrieval becomes noisy
because the embedding averages over unrelated content.

**This project's approach:** 512-character chunks with 64-char overlap and paragraph-first
splitting. Overlap preserves cross-boundary context.

## Q: How do embeddings work in this implementation?

`sentence-transformers/all-MiniLM-L6-v2` encodes text to 384-dimensional vectors using
a fine-tuned transformer. The model is trained with contrastive learning to place
semantically similar texts near each other in vector space.

We normalize vectors to unit length so that cosine similarity equals dot product.
Both query and document embeddings use the same model and normalization — inconsistency
here breaks retrieval quality.

## Q: When does BM25 outperform dense retrieval?

BM25 outperforms when queries use exact terminology that the embedding model has not
generalized well:
- Acronyms: "RFC 4271", "BERT", "GPT-4"
- Named entities with unusual tokenization
- Technical identifiers like model names, part numbers, error codes
- New terminology introduced after the embedding model's training cutoff

In this project, hybrid retrieval combines both via RRF to get the benefits of both.

## Q: How does Reciprocal Rank Fusion work?

For each document d and each result list r:
```
rrf_score(d) = Σ_r  1 / (k + rank_r(d))
```
where k=60 is a smoothing constant.

Documents appearing near the top of multiple lists accumulate high scores.
Documents in only one list get a lower score.

Crucially, RRF uses only ranks — not raw scores — so it works correctly when
combining BM25 (unbounded scores) and cosine similarity (0 to 1).

## Q: What problem does reranking solve?

First-stage retrieval (dense or sparse) uses approximate similarity for speed.
Cross-encoders jointly encode the query and each passage together, making
fine-grained relevance distinctions the bi-encoder missed.

In this project: Stage 1 retrieves `candidate_k=20` results in ~45ms.
Stage 2 reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2` in ~200-400ms.
Final output is the top-k reranked results.

## Q: What is the difference between MMR and reranking?

**MMR** optimizes for diversity: given already-selected results, the next result
should be relevant to the query but dissimilar to what was already selected.
MMR does not improve relevance — it trades some relevance for coverage.

**Reranking** optimizes for relevance: a cross-encoder re-scores candidates by
relevance to the query. It may return near-duplicate results if they are all relevant.

Use MMR when: the top results are near-duplicates and you want varied perspectives.
Use reranking when: you need the most precisely relevant results regardless of diversity.

## Q: Why can a correct retrieval pipeline still produce a bad answer?

Multiple failure modes exist independently:

1. **Retrieval precision:** Wrong documents retrieved — nothing relevant in context.
2. **Retrieval recall:** Right documents not retrieved — relevant chunk ranked low.
3. **Generation faithfulness:** Correct context retrieved but LLM adds facts not in evidence.
4. **Citation accuracy:** LLM attributes information to the wrong source.
5. **Prompt injection:** Adversarial document content overrides instructions.

This project separates retrieval and generation so failures can be diagnosed separately.
The Retrieval Explorer shows what was retrieved. The answer shows what was generated.

## Q: How does HyDE work and what is its risk?

HyDE (Hypothetical Document Embeddings):
1. LLM generates a hypothetical answer paragraph for the query
2. Hypothetical paragraph is embedded (not the original query)
3. Real documents are retrieved using the hypothesis embedding
4. Final answer is generated from REAL retrieved documents (hypothesis discarded)

**Risk:** If the LLM hallucfinates in the hypothesis, the retrieval may chase false terms.
**Mitigation:** The hypothesis is never shown to the user or used as evidence.

## Q: How does image retrieval work?

1. Images are extracted from documents or provided standalone
2. A vision model generates a text description of each image
3. The description is embedded and stored in the visual ChromaDB collection
4. The original image bytes are stored in metadata (as base64)
5. At query time, text queries match against image descriptions
6. Retrieved images are decoded and sent to the vision model with the user's question
7. The vision model produces an answer grounded in the actual image

This ensures the final answer comes from real visual evidence, not just the text description.

## Q: How are page-level citations generated?

Each text chunk is created with a `page_number` field from the loader.
This metadata is stored in ChromaDB and retrieved with the chunk.
The `format_context()` function formats chunks as `[N] Source: X, Page: Y`.
`validate_citations()` checks that every `[N]` in the answer corresponds
to an actual retrieved source with a real page number.

## Q: How do you detect retrieval failures?

- **Hit@k = 0:** The expected source was not in the top-k results.
- **MRR < 0.3:** The first relevant result is typically ranked 3+ positions down.
- **Recall@k < 0.5:** Less than half of the expected sources were retrieved.

The benchmark runner (`scripts/evaluate_retrieval.py`) computes these for each config.

## Q: How would you improve latency?

1. **Reduce candidate_k:** Fewer candidates for BM25 and reranking.
2. **Disable reranking:** Remove the cross-encoder stage.
3. **Use a smaller embedding model:** all-MiniLM-L6-v2 is already small; could use distillbert.
4. **HNSW tuning:** ChromaDB uses HNSW; tune `ef_construction` and `ef_search`.
5. **Cache frequent queries:** LRU cache on embeddings and results.
6. **Stream responses:** Start showing the answer before generation completes.

## Q: How would you evaluate a change to the chunking strategy?

1. Keep the evaluation dataset fixed (do not change questions or expected sources).
2. Re-ingest documents with the new chunking parameters.
3. Re-run `scripts/evaluate_retrieval.py` for all configurations.
4. Compare Hit@k, Recall@k, MRR, nDCG@k before and after.
5. Also check answer quality for cases where retrieval improved.

Do not evaluate by asking "does it feel better?" — measure it.

## Q: What would you change before deploying to multiple users?

1. Add authentication (e.g., JWT tokens)
2. Per-user document isolation (separate ChromaDB collections)
3. Rate limiting on API endpoints
4. Request queuing for concurrent embedding/LLM calls
5. Replace local JSON document store with a proper database (PostgreSQL)
6. Move BM25 to a background rebuild worker
7. Add proper error monitoring (Sentry or similar)
8. Replace print/log to file with a log aggregation system
9. Implement proper file storage (S3 or similar) instead of local disk
10. Add a proper API gateway for TLS termination
