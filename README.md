🧠 Multimodal RAG Knowledge Assistant

A document-intelligence application for asking questions across text, tables, charts, diagrams, and images in a document collection. The project combines Retrieval-Augmented Generation (RAG) with hybrid search, optional reranking, grounded citations, retrieval evaluation, a FastAPI backend, and a Streamlit interface.

Project status: The codebase includes the components and test suite described below. Run the setup and verification steps in your environment before treating every feature as end-to-end verified. No benchmark scores or performance improvements are claimed in this README.

📌 Project Overview

A basic document chatbot can struggle when documents contain many pages, tables, and visual elements. Sending entire documents to an LLM can also add irrelevant context and make it harder to trace an answer back to its source.

This project follows a retrieval-first approach: documents are processed and indexed, relevant evidence is retrieved for a user query, and the answer-generation stage uses that evidence to produce a response with source references.

🎯 Key Objectives

Build a modular ingestion and retrieval pipeline for multiple document formats.

Combine dense semantic retrieval with sparse keyword retrieval.

Improve retrieval relevance through rank fusion and optional refinement strategies.

Support text, tables, page renders, and image-oriented retrieval workflows.

Generate answers with source-grounded citations.

Evaluate retrieval quality using standard ranking metrics.

Expose the application through a REST API and an interactive web interface.

Keep local development, configuration, and containerization documented and reproducible.

🚀 Features

1. 📄 Multi-format Document Ingestion

Supports PDF, DOCX, TXT, Markdown, CSV, PPTX, PNG, and JPG/JPEG inputs in the document-loading code.

Extracts text and structured tables where supported by the source format.

Processes PDF page renders and embedded visual content for visual retrieval workflows.

Uses chunking and metadata to prepare content for retrieval.

Includes document-processing and ingestion pipeline modules.

2. 🔎 Hybrid Information Retrieval

Dense retrieval: Uses Sentence Transformers embeddings with ChromaDB for semantic similarity search.

Sparse retrieval: Uses BM25 for keyword-based matching.

Rank fusion: Combines dense and sparse result rankings with Reciprocal Rank Fusion (RRF).

Candidate retrieval: Supports retrieving a broader candidate set before optional refinement.

3. 🖼️ Multimodal and Visual Retrieval

Creates searchable descriptions of visual content through the configured vision provider.

Includes a visual retriever for searching indexed visual descriptions.

Supports workflows involving document page renders and embedded images.

Uses a configurable local Ollama vision model in the example environment configuration.

Vision behavior depends on the selected provider, installed model, and available hardware. Verify image-question-answering workflows locally before describing them as fully operational.

4. 🧭 Query Enhancement

The codebase includes modules for several query-processing strategies:

Query rewriting

Query expansion

Multi-query retrieval

Query decomposition

Hypothetical Document Embeddings (HyDE)

Follow-up query reformulation for conversational context

These strategies can help adapt a user question before or during retrieval. Their effectiveness depends on the model configuration and the query being asked.

5. 🎯 Retrieval Refinement

Cross-encoder reranking: Optional second-stage ranking of retrieved candidates.

Maximum Marginal Relevance (MMR): Optional diversification of results to balance relevance and diversity.

Configuration flags allow these refinement stages to be enabled or disabled.

6. 🧾 Grounded Answer Generation

Generates answers from retrieved context through a configurable LLM provider.

Includes citation formatting and validation components.

Supports a local Ollama text-generation configuration by default in .env.example.

Keeps provider configuration separate from the RAG pipeline code.

Citations improve traceability, but they do not guarantee that every generated statement is correct. Verify important answers against the cited documents.

7. 📊 Retrieval Evaluation

The project includes an evaluation dataset and code for retrieval metrics such as:

Metric

What it measures

Hit@k

Whether a relevant item appears among the top k results.

Recall@k

The fraction of known relevant items found among the top k results.

MRR

The reciprocal rank of the first relevant result, averaged across queries.

nDCG@k

Ranking quality that accounts for the position and relevance of results.

Metric values depend on the evaluation dataset, relevance labels, retrieval configuration, and model settings. Add numerical results only after running and documenting a reproducible evaluation.

8. 🌐 API and Interactive Interface

FastAPI backend: REST endpoints, request/response schemas, and interactive API documentation.

Streamlit frontend: Interactive application with pages for document question answering and retrieval/evaluation workflows.

Docker support: The repository includes a Dockerfile and docker-compose.yml for container-based experimentation.

🏗️ Project Architecture

The application is organized around document ingestion, indexing, retrieval, and grounded generation.

                         DOCUMENT INGESTION
  PDF / DOCX / TXT / MD / CSV / PPTX / PNG / JPG
                            │
                            ▼
                 Loaders + Document Processing
                            │
                            ▼
                     Chunking + Metadata
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
       Dense Embeddings            Sparse Index
       Sentence Transformers       BM25
               │                         │
               ▼                         ▼
            ChromaDB                BM25 Retrieval
               └────────────┬────────────┘
                            ▼
                  Hybrid Rank Fusion (RRF)
                            │
                 Optional MMR / Reranking
                            │
                            ▼
                     Relevant Context
                            │
                 Configured LLM Provider
                            │
                            ▼
                 Answer + Source Citations

  Visual path: Images / Page Renders → Vision Descriptions → Visual Retrieval

  Access layer: Streamlit UI ↔ FastAPI Backend

Main Pipeline Stages

Ingestion: Load supported documents and extract available text, tables, and visual content.

Processing: Split extracted content into chunks and attach metadata.

Indexing: Create embeddings and maintain dense and sparse retrieval indexes.

Query processing: Optionally rewrite, expand, decompose, or generate variants of a question.

Retrieval: Search the dense and sparse indexes and combine rankings.

Refinement: Optionally diversify or rerank the candidate results.

Generation: Assemble context and generate an answer with source citations.

Evaluation: Use the supplied dataset and retrieval metrics to inspect search quality.

🧰 Technology Stack

Category

Technologies

Language

Python 3.11+

API

FastAPI, Uvicorn, Pydantic

UI

Streamlit

Embeddings

Sentence Transformers, all-MiniLM-L6-v2

Vector database

ChromaDB

Sparse retrieval

rank-bm25 / BM25Okapi

Reranking

Cross-encoder model, optional

Query strategies

Rewrite, expansion, multi-query, decomposition, HyDE

LLM / vision providers

Ollama by default in the example configuration; provider code is configurable

Document processing

pdfplumber, python-docx, python-pptx, Pillow, CSV utilities

Evaluation

Hit@k, Recall@k, MRR, nDCG@k and answer-metric utilities

Testing

pytest

Packaging

Docker, Docker Compose

Development tools

Git, GitHub, VS Code

📂 Project Structure

multimodal-rag-knowledge-assistant/
├── app/
│   ├── api/                  # FastAPI application, routes, schemas, dependencies
│   ├── chunking/             # Text chunking and metadata handling
│   ├── config/               # Settings and logging configuration
│   ├── embeddings/           # Embedding generation
│   ├── evaluation/           # Retrieval and answer evaluation
│   ├── generation/           # LLM/vision providers, prompts, answers, citations
│   ├── ingestion/            # Document loading and processing
│   ├── pipeline/             # Ingestion and RAG orchestration
│   ├── query_processing/     # Rewrite, expansion, decomposition, HyDE, multi-query
│   ├── retrieval/             # Dense, sparse, hybrid, visual, MMR, and reranking
│   ├── services/             # Application services
│   └── storage/              # Document stores, manifests, and vector storage
├── data/
│   ├── demo_corpus/           # Synthetic sample documents for demonstrations
│   └── evaluation/            # Retrieval evaluation dataset
├── docs/                      # Architecture, design decisions, guides, limitations
├── frontend/
│   └── streamlit_app.py       # Streamlit application
├── scripts/                   # Corpus, ingestion, query, and evaluation utilities
├── tests/
│   ├── api/
│   ├── integration/
│   └── unit/                  # Unit tests for core components
├── .env.example               # Safe environment-variable template
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md

Generated vector-store data and uploaded documents under data/chroma_db/ and data/uploads/ should remain local and are excluded from version control by the project’s .gitignore rules.

⚙️ Installation and Setup

Prerequisites

Python 3.11 is recommended because the project configuration requires Python 3.11 or later and several dependencies are pinned to specific versions.

Git.

Ollama, if using the configured local text or vision model providers.

Docker Desktop, if trying the containerized setup.

The first run may download Python packages and machine-learning models. Model loading and inference time depend on model size, available memory, hardware, and provider configuration.

Step 1: Clone the Repository

After creating the GitHub repository, use its actual URL. For the planned repository name, the command will be:

git clone https://github.com/Harshitraj123/multimodal-rag-knowledge-assistant.git
cd multimodal-rag-knowledge-assistant

Step 2: Create a Virtual Environment

In Windows PowerShell:

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

If the py -3.11 command is unavailable, install Python 3.11 or use another compatible Python version after checking dependency compatibility. Avoid changing the system-wide execution policy just to activate this project’s environment.

Step 3: Install Dependencies

python -m pip install -r requirements.txt

Step 4: Create the Local Environment File

Copy the example configuration to a private .env file:

Copy-Item .env.example .env

Review .env and adjust values to match your local setup. The example configuration uses the following defaults:

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2
VISION_PROVIDER=ollama
VISION_MODEL=llava
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

Do not replace real credentials with placeholders in your local .env; keep actual credentials private. Never commit .env to GitHub. Only publish .env.example with non-secret example values.

Step 5: Prepare Ollama Models (Local Inference)

If Ollama is your selected provider, install and start Ollama first. Then pull the configured models:

ollama pull llama3.2
ollama pull llava

The commands above are relevant only when using the corresponding local text and vision models. Model availability and resource requirements depend on your machine. If you choose another provider, configure the environment variables expected by the provider implementation instead.

▶️ Running the Application

Run the backend and frontend in separate terminals from the project root, with the virtual environment activated in each terminal.

1. Start the FastAPI Backend

uvicorn app.api.main:app --reload

When startup succeeds, open the interactive API documentation at:

http://127.0.0.1:8000/docs

2. Start the Streamlit Frontend

In a second terminal:

streamlit run frontend/streamlit_app.py

Open the local URL printed by Streamlit. If the interface cannot communicate with the backend, verify that the API process is running and review the API address configured in the application settings.

3. Demo, Ingestion, and Evaluation Scripts

The repository includes helper scripts for creating a demo corpus, ingesting documents, running a query demonstration, and evaluating retrieval. Inspect their options or configuration before running them:

python scripts/create_demo_corpus.py
python scripts/ingest.py
python scripts/query_demo.py
python scripts/evaluate_retrieval.py

Run the scripts only when their actions are appropriate for your local data. In particular, verify the configured data paths and persistent-store behavior before re-ingesting a corpus.

🧪 Testing

Run the full test suite from the project root:

python -m pytest

Run unit tests separately:

python -m pytest tests/unit -v

A passing unit-test suite is useful evidence for the components covered by those tests, but it does not by itself prove that the complete app works with live Ollama models, uploaded documents, Docker, and every UI workflow. Record the test command and result after executing it in your environment.

🐳 Docker

The repository includes a Dockerfile and docker-compose.yml. To try the container configuration:

docker compose up --build

Review the Compose file and required environment variables before starting containers. Local model access may need additional configuration so that a container can reach the Ollama service on the host. Container startup and model connectivity should be verified in the target environment.

Stop a running Compose session with Ctrl+C; use docker compose down when appropriate.

📊 Evaluation and Quality Checks

The project includes retrieval-evaluation utilities and a version-controlled evaluation dataset. To make results reproducible, report:

The evaluation dataset version and number of queries.

The retrieval strategy and values for TOP_K and CANDIDATE_K.

Whether query enhancement, MMR, or reranking was enabled.

The embedding, reranker, and LLM model names used.

Hit@k, Recall@k, MRR, and nDCG@k values from the actual run.

No performance figures are reported here because they should come from a measured run, not estimates.

🔄 Development Workflow and Status

Component

Purpose

Status in this repository

Document ingestion

Load supported formats and prepare content

Code modules present; verify end-to-end locally

Dense and sparse retrieval

Semantic and keyword search

Retrieval modules present; verify against the evaluation dataset

Rank fusion

Combine dense and sparse rankings

Fusion module and unit tests present

Query enhancement

Rewrite, expand, decompose, multi-query, and HyDE

Strategy modules present; verify with configured providers

Generation and citations

Build evidence-grounded answers

Provider and citation modules present; verify with live models

FastAPI backend

Expose application endpoints

Application and route modules present; test startup and endpoints

Streamlit frontend

Provide an interactive UI

Frontend file present; test UI/backend integration

Evaluation

Measure retrieval and answer quality

Evaluation modules and dataset present; run and record results

Docker

Containerized execution

Configuration files present; build and test containers

🛣️ Future Enhancements

Potential next steps for the project include:

Add a reproducible benchmark report with dataset details and measured retrieval results.

Expand integration tests for live provider connectivity and document-upload workflows.

Add automated linting and test execution to a GitHub Actions workflow.

Improve provider error messages and health checks for local model services.

Add deployment-specific configuration and documented operational monitoring.

Capture response latency and resource use under a defined test setup.

Document a repeatable demonstration using the included synthetic corpus.

These are suggested improvements, not claims that the items are already implemented.

🔐 Security and Data Handling

Keep .env, API keys, access tokens, and private connection strings out of version control.

Publish .env.example with safe, non-secret values only.

Do not commit private PDFs, customer documents, uploaded files, generated vector databases, or other local artifacts unless they are intentionally public and appropriately licensed.

Review .gitignore before staging files and inspect the staged diff before every commit.

Treat retrieved documents as evidence to check, not as a guarantee that an LLM response is correct.

Review docs/LIMITATIONS.md before relying on generated answers in consequential situations.

🎓 Learning Outcomes

This project provides hands-on work with:

Retrieval-Augmented Generation (RAG) system structure.

Multiformat document ingestion and chunking.

Embeddings, vector stores, and sparse retrieval.

Hybrid search and Reciprocal Rank Fusion.

Reranking and diversity-aware retrieval.

Query enhancement patterns for LLM applications.

Source-grounded generation and citation handling.

Retrieval evaluation metrics and testable components.

FastAPI, Streamlit, environment configuration, and Docker basics.

👨‍💻 Author

Harshit Raj
Computer Science & Engineering | AI/ML | Generative AI

GitHub: Harshitraj123

LinkedIn: harshitraj010204

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgment

Architecture concepts were informed in part by techwithprateek/agenticai-projects. This repository presents its own code organization, retrieval pipeline, tests, evaluation utilities, and interface; retain this acknowledgment when publishing if it accurately reflects the project's inspiration.