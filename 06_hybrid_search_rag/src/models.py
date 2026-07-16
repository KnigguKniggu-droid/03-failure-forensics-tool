"""Typed contracts for the hybrid search RAG pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentSource(str, Enum):
    MARKDOWN = "markdown"
    PYTHON = "python"
    JSON = "json"
    TEXT = "text"
    PDF = "pdf"


class Document(BaseModel):
    """A parsed document chunk stored in the vector index."""

    doc_id: str
    content: str = Field(..., min_length=1)
    source_file: str
    source_type: DocumentSource
    chunk_index: int = Field(..., ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    content_hash: str = Field(..., description="SHA-256 hash for deduplication")


class RetrievalResult(BaseModel):
    """A single retrieval result from either BM25 or vector search."""

    doc_id: str
    content: str
    source_file: str
    score: float
    rank: int
    retrieval_method: str = Field(..., description="bm25 | vector | fused")
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusedResult(BaseModel):
    """Result after Reciprocal Rank Fusion of BM25 and vector scores."""

    doc_id: str
    content: str
    source_file: str
    rrf_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    final_rank: int


class RerankedResult(BaseModel):
    """Result after cross-encoder reranking."""

    doc_id: str
    content: str
    source_file: str
    cross_encoder_score: float
    rrf_rank: int
    final_rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationSpan(BaseModel):
    """A bracketed citation found in generated text."""

    citation_text: str = Field(..., description="The bracketed citation text")
    start_pos: int
    end_pos: int
    cited_doc_id: str | None = Field(None, description="Matched document ID if verified")


class CitationVerification(BaseModel):
    """Result of citation verification on a generated response."""

    total_citations: int
    verified_citations: int
    unverified_citations: int
    hallucinated_citations: int
    citation_spans: list[CitationSpan] = Field(default_factory=list)
    verification_score: float = Field(..., ge=0.0, le=1.0)
    details: list[str] = Field(default_factory=list)


class RAGRequest(BaseModel):
    """API request for RAG query."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(20, ge=1, le=100, description="Number of results to retrieve before reranking")
    rerank_top: int = Field(5, ge=1, le=20, description="Number of results to return after reranking")


class RAGResponse(BaseModel):
    """API response from the RAG pipeline."""

    query: str
    answer: str
    retrieved_docs: list[RerankedResult]
    citations: CitationVerification
    total_latency_ms: float
