"""
Pydantic request/response schemas for the FastAPI API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator

# Type Aliases for strict schema constraints & Swagger dropdowns
RetrievalMethod = Literal["dense", "sparse", "hybrid"]
QueryStrategy = Literal["none", "rewrite", "expand", "multi_query", "decompose", "hyde"]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    capabilities: dict[str, Any]
    indexed_documents: int
    indexed_chunks: int


class UploadResponse(BaseModel):
    file_name: str
    saved_path: str
    size_bytes: int


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to ingest (server-side)")

    @field_validator("file_path")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        path = Path(v)
        # Prevent path traversal and absolute system path access
        if ".." in v or path.is_absolute():
            raise ValueError("Invalid path: path traversal or absolute system paths not allowed")
        return v


class IngestResponse(BaseModel):
    doc_id: str
    file_name: str
    status: str
    chunks_added: int
    tables_indexed: int
    visuals_indexed: int
    extraction_errors: list[str]
    processing_time_s: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    retrieval_method: RetrievalMethod = Field("hybrid", description="dense | sparse | hybrid")
    query_strategy: QueryStrategy = Field("none", description="none | rewrite | expand | multi_query | decompose | hyde")
    top_k: int = Field(5, ge=1, le=20)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class SourceInfo(BaseModel):
    citation_num: int
    source: str
    page_number: Optional[Union[int, str]] = None
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    citation_warnings: list[str]
    query_used: str
    original_query: str
    query_strategy: str
    retrieval_method: str
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    evidence_sufficient: bool
    retrieval_results_count: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    retrieval_method: RetrievalMethod = Field("hybrid", description="dense | sparse | hybrid")
    top_k: int = Field(10, ge=1, le=50)
    include_scores: bool = True


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    page_number: Optional[Union[int, str]] = None
    score: float
    retrieval_method: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    retrieval_method: str
    latency_s: float
    total_results: int


class DocumentInfo(BaseModel):
    doc_id: str
    file_name: str
    file_type: str
    chunk_count: int = 0
    table_count: int = 0
    visual_count: int = 0
    ingested_at: Optional[float] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None