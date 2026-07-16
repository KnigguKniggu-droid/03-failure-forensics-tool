"""Hybrid retriever combining BM25 and vector search with RRF.

Runs BM25 and vector retrieval in parallel, fuses the results using
Reciprocal Rank Fusion, and returns the top_k fused results.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from src.models import Document, FusedResult, RetrievalResult

RRF_K = 60  # RRF constant (standard value from the original paper)


class BM25Retriever:
    """BM25 keyword retrieval over document corpus."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.corpus = [doc.content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus) if HAS_BM25 and self.corpus else None

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievalResult] = []
        for rank, idx in enumerate(ranked_indices):
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            results.append(RetrievalResult(
                doc_id=doc.doc_id,
                content=doc.content,
                source_file=doc.source_file,
                score=float(scores[idx]),
                rank=rank + 1,
                retrieval_method="bm25",
            ))
        return results


class VectorRetriever:
    """Vector similarity retrieval using cosine similarity."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = [d for d in documents if d.embedding is not None]

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[RetrievalResult]:
        if not self.documents:
            return []
        query_vec = np.array(query_embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scores: list[tuple[int, float]] = []
        for i, doc in enumerate(self.documents):
            doc_vec = np.array(doc.embedding, dtype=np.float64)
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                continue
            sim = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(scores[:top_k]):
            doc = self.documents[idx]
            results.append(RetrievalResult(
                doc_id=doc.doc_id,
                content=doc.content,
                source_file=doc.source_file,
                score=score,
                rank=rank + 1,
                retrieval_method="vector",
            ))
        return results


def reciprocal_rank_fusion(
    bm25_results: list[RetrievalResult],
    vector_results: list[RetrievalResult],
    top_k: int = 20,
    k: int = RRF_K,
) -> list[FusedResult]:
    """Fuse BM25 and vector results using Reciprocal Rank Fusion.

    RRF score = 1 / (k + rank) for each retrieval method, summed.
    """
    rrf_scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    vector_ranks: dict[str, int] = {}
    doc_data: dict[str, Document | RetrievalResult] = {}

    for result in bm25_results:
        rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0.0) + 1.0 / (k + result.rank)
        bm25_ranks[result.doc_id] = result.rank
        doc_data[result.doc_id] = result

    for result in vector_results:
        rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0.0) + 1.0 / (k + result.rank)
        vector_ranks[result.doc_id] = result.rank
        doc_data[result.doc_id] = result

    sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)[:top_k]

    fused: list[FusedResult] = []
    for rank, doc_id in enumerate(sorted_ids):
        data = doc_data[doc_id]
        fused.append(FusedResult(
            doc_id=doc_id,
            content=data.content,
            source_file=data.source_file,
            rrf_score=rrf_scores[doc_id],
            bm25_rank=bm25_ranks.get(doc_id),
            vector_rank=vector_ranks.get(doc_id),
            final_rank=rank + 1,
        ))
    return fused


class HybridRetriever:
    """Parallel hybrid retriever combining BM25 and vector search."""

    def __init__(self, documents: list[Document]) -> None:
        self.bm25 = BM25Retriever(documents)
        self.vector = VectorRetriever(documents)
        self.documents = documents

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 20,
    ) -> list[FusedResult]:
        bm25_results = self.bm25.search(query, top_k=top_k)
        vector_results = (
            self.vector.search(query_embedding, top_k=top_k)
            if query_embedding is not None
            else []
        )
        return reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k)
