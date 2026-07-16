"""FastAPI application for the Hybrid Search RAG pipeline."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models import Document, RAGRequest, RAGResponse
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.citation_verifier import verify_citations
from src.ingestion import ingest_directory, deduplicate
from pathlib import Path

app = FastAPI(
    title="Hybrid Search RAG Pipeline",
    description="BM25 + Vector Fusion RAG with citation verification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: HybridRetriever | None = None
_reranker = CrossEncoderReranker()


@app.post("/v1/ingest")
async def ingest_docs(directory: str) -> dict[str, Any]:
    """Ingest documents from a directory into the retriever index."""
    global _retriever
    docs = ingest_directory(Path(directory))
    docs = deduplicate(docs)
    _retriever = HybridRetriever(docs)
    return {"ingested": len(docs), "duplicates_removed": len(ingest_directory(Path(directory))) - len(docs)}


@app.post("/v1/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest) -> RAGResponse:
    """Run a hybrid search RAG query with reranking and citation verification."""
    if _retriever is None:
        raise HTTPException(status_code=400, detail="No documents ingested. Call /v1/ingest first.")

    start = time.monotonic()
    fused = _retriever.search(request.query, top_k=request.top_k)
    reranked = _reranker.rerank(request.query, fused, top_n=request.rerank_top)

    context = "\n\n".join(f"[{doc.doc_id}] {doc.content}" for doc in reranked)
    answer = f"Based on the retrieved documents:\n\n{context}"

    citations = verify_citations(answer, reranked)
    latency = (time.monotonic() - start) * 1000

    return RAGResponse(
        query=request.query,
        answer=answer,
        retrieved_docs=reranked,
        citations=citations,
        total_latency_ms=latency,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
