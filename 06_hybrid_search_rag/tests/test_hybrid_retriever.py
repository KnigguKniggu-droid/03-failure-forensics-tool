"""Tests for the hybrid retriever, RRF fusion, and citation verifier."""

from __future__ import annotations

import pytest

from src.models import Document, DocumentSource, FusedResult, RetrievalResult
from src.hybrid_retriever import reciprocal_rank_fusion, BM25Retriever, HybridRetriever
from src.citation_verifier import extract_citations, verify_citations
from src.reranker import CrossEncoderReranker
from src.models import RerankedResult
from src.ingestion import chunk_text, hash_content, deduplicate


def _make_doc(doc_id: str, content: str, embedding: list[float] | None = None) -> Document:
    return Document(
        doc_id=doc_id,
        content=content,
        source_file=f"test/{doc_id}.txt",
        source_type=DocumentSource.TEXT,
        chunk_index=0,
        content_hash=hash_content(content),
        embedding=embedding,
    )


def test_chunk_text_produces_overlapping_chunks():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_rrf_fuses_bm25_and_vector_results():
    bm25 = [
        RetrievalResult(doc_id="d1", content="a", source_file="f1", score=5.0, rank=1, retrieval_method="bm25"),
        RetrievalResult(doc_id="d2", content="b", source_file="f2", score=3.0, rank=2, retrieval_method="bm25"),
    ]
    vector = [
        RetrievalResult(doc_id="d2", content="b", source_file="f2", score=0.9, rank=1, retrieval_method="vector"),
        RetrievalResult(doc_id="d3", content="c", source_file="f3", score=0.8, rank=2, retrieval_method="vector"),
    ]
    fused = reciprocal_rank_fusion(bm25, vector, top_k=10)
    assert len(fused) == 3
    ids = {f.doc_id for f in fused}
    assert ids == {"d1", "d2", "d3"}
    d2 = next(f for f in fused if f.doc_id == "d2")
    assert d2.bm25_rank == 2
    assert d2.vector_rank == 1


def test_bm25_retriever_returns_ranked_results():
    docs = [
        _make_doc("d1", "the quick brown fox"),
        _make_doc("d2", "the lazy dog sleeps"),
        _make_doc("d3", "quick fox jumps over dog"),
    ]
    retriever = BM25Retriever(docs)
    results = retriever.search("quick fox", top_k=2)
    assert len(results) <= 2
    assert all(r.retrieval_method == "bm25" for r in results)


def test_deduplicate_removes_near_duplicates():
    docs = [
        _make_doc("d1", "hello world", embedding=[1.0, 0.0, 0.0]),
        _make_doc("d2", "hello world", embedding=[0.99, 0.01, 0.0]),
        _make_doc("d3", "completely different", embedding=[0.0, 0.0, 1.0]),
    ]
    deduped = deduplicate(docs, threshold=0.95)
    assert len(deduped) == 2
    assert "d3" in {d.doc_id for d in deduped}


def test_extract_citations_finds_bracketed_refs():
    text = "According to [doc:readme:0], the API is stable. See [doc:guide:3] for details."
    citations = extract_citations(text)
    assert len(citations) == 2
    assert citations[0].citation_text == "doc:readme:0"
    assert citations[1].citation_text == "doc:guide:3"


def test_verify_citations_flags_hallucinated():
    docs = [
        RerankedResult(doc_id="d1", content="real content here", source_file="f1.txt",
                       cross_encoder_score=0.9, rrf_rank=1, final_rank=1),
    ]
    text = "Based on [d1] and [fake_doc]."
    result = verify_citations(text, docs)
    assert result.total_citations == 2
    assert result.verified_citations >= 1
    assert result.hallucinated_citations >= 1


def test_cross_encoder_reranker_orders_by_relevance():
    fused = [
        FusedResult(doc_id="d1", content="the quick brown fox", source_file="f1",
                    rrf_score=0.5, final_rank=1),
        FusedResult(doc_id="d2", content="unrelated content about cooking", source_file="f2",
                    rrf_score=0.4, final_rank=2),
        FusedResult(doc_id="d3", content="quick fox jumps over the dog", source_file="f3",
                    rrf_score=0.3, final_rank=3),
    ]
    reranker = CrossEncoderReranker()
    results = reranker.rerank("quick fox", fused, top_n=2)
    assert len(results) == 2
    assert results[0].cross_encoder_score >= results[1].cross_encoder_score
