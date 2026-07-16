"""Cross-encoder reranker for the top fused results.

Routes the highest 20 RRF results through a cross-encoder model
that jointly encodes the query and document for fine-grained relevance scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models import FusedResult, RerankedResult


class CrossEncoderReranker:
    """Cross-encoder reranker using a joint query-document scoring model.

    In production, this would load a pre-trained cross-encoder model
    (e.g., ms-marco-MiniLM-L-6-v2). For the architectural prototype,
    a lightweight scoring function is used.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            self._model = None

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        if self._model is not None:
            return float(self._model.predict([(query, document)])[0])
        query_tokens = set(query.lower().split())
        doc_tokens = set(document.lower().split())
        if not doc_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens)
        jaccard = overlap / len(query_tokens | doc_tokens) if (query_tokens | doc_tokens) else 0.0
        return jaccard

    def rerank(
        self,
        query: str,
        fused_results: list[FusedResult],
        top_n: int = 5,
    ) -> list[RerankedResult]:
        """Rerank fused results using cross-encoder scoring."""
        if not fused_results:
            return []

        scored: list[tuple[float, FusedResult]] = []
        for result in fused_results:
            score = self.score_pair(query, result.content)
            scored.append((score, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked: list[RerankedResult] = []
        for rank, (score, result) in enumerate(scored[:top_n]):
            reranked.append(RerankedResult(
                doc_id=result.doc_id,
                content=result.content,
                source_file=result.source_file,
                cross_encoder_score=score,
                rrf_rank=result.final_rank,
                final_rank=rank + 1,
            ))
        return reranked
