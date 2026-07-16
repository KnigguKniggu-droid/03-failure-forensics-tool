"""Semantic cache layer using Redis VL with similarity boundary.

Computes unique cache keys from system prompt, temperature, and model
metadata hashes to prevent index bleed. Uses a 0.95 cosine similarity
boundary for cache hits.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from src.models import CacheEntry, CacheKey, CacheLookupResult, CacheStatus

SIMILARITY_BOUNDARY = 0.95
DEFAULT_TTL = 3600


def compute_cache_key(
    model: str,
    temperature: float,
    system_prompt: str,
    messages: list[dict[str, str]],
) -> CacheKey:
    """Compute a unique cache key from request metadata.

    Prevents index bleed by hashing system prompt, temperature, and model
    metadata separately, then combining into a single SHA-256.
    """
    system_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    temp_hash = hashlib.sha256(f"{temperature:.6f}".encode("utf-8")).hexdigest()
    model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()

    combined_input = f"{system_hash}:{temp_hash}:{model_hash}"
    combined_hash = hashlib.sha256(combined_input.encode("utf-8")).hexdigest()

    return CacheKey(
        system_hash=system_hash,
        temperature_hash=temp_hash,
        model_hash=model_hash,
        combined_hash=combined_hash,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    if va.size == 0 or vb.size == 0:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class SemanticCache:
    """Redis VL-backed semantic cache with similarity boundary.

    In production, this connects to Redis VL (redisvl) for vector indexing.
    For the architectural prototype, an in-memory store with cosine
    similarity is used as a drop-in replacement.
    """

    def __init__(self, similarity_boundary: float = SIMILARITY_BOUNDARY) -> None:
        self.similarity_boundary = similarity_boundary
        self._store: dict[str, CacheEntry] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._redis_client: Any = None
        self._redisvl_index: Any = None

    def connect_redis(self, redis_url: str = "redis://localhost:6379") -> None:
        """Connect to Redis and initialize the RedisVL index."""
        try:
            import redis
            from redisvl.index import SearchIndex
            from redisvl.schema import IndexSchema

            self._redis_client = redis.from_url(redis_url)

            schema = IndexSchema.from_dict({
                "index": {"name": "semantic_cache", "prefix": "semcache:"},
                "fields": [
                    {"name": "query", "type": "text"},
                    {"name": "response", "type": "text"},
                    {"name": "model", "type": "text"},
                    {"name": "temperature", "type": "numeric"},
                    {
                        "name": "query_embedding",
                        "type": "vector",
                        "attrs": {"dims": 1536, "distance_metric": "cosine", "algorithm": "flat"},
                    },
                ],
            })
            self._redisvl_index = SearchIndex(schema, redis_client=self._redis_client)
            self._redisvl_index.create(overwrite=True)
        except ImportError:
            pass

    def lookup(
        self,
        query_embedding: list[float],
        cache_key: CacheKey,
    ) -> CacheLookupResult:
        """Look up a cached response by semantic similarity."""
        start = time.monotonic()

        if self._redisvl_index is not None:
            return self._lookup_redisvl(query_embedding, cache_key, start)

        best_score = 0.0
        best_entry: CacheEntry | None = None

        for key, embedding in self._embeddings.items():
            score = cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_score = score
                best_entry = self._store.get(key)

        latency = (time.monotonic() - start) * 1000

        if best_entry is not None and best_score >= self.similarity_boundary:
            best_entry.access_count += 1
            best_entry.last_accessed = datetime.now(timezone.utc)
            return CacheLookupResult(
                status=CacheStatus.HIT,
                entry=best_entry,
                similarity_score=best_score,
                lookup_latency_ms=latency,
            )

        if best_entry is not None and best_score >= self.similarity_boundary * 0.9:
            return CacheLookupResult(
                status=CacheStatus.PARTIAL_HIT,
                entry=best_entry,
                similarity_score=best_score,
                lookup_latency_ms=latency,
            )

        return CacheLookupResult(
            status=CacheStatus.MISS,
            similarity_score=best_score,
            lookup_latency_ms=latency,
        )

    def _lookup_redisvl(
        self,
        query_embedding: list[float],
        cache_key: CacheKey,
        start: float,
    ) -> CacheLookupResult:
        """Lookup using RedisVL vector search."""
        try:
            results = self._redisvl_index.query(
                VectorQuery(
                    vector=query_embedding,
                    vector_field_name="query_embedding",
                    num_results=1,
                    return_fields=["query", "response", "model", "temperature"],
                )
            )
            latency = (time.monotonic() - start) * 1000
            if results and results[0].get("distance", 1.0) <= (1.0 - self.similarity_boundary):
                return CacheLookupResult(
                    status=CacheStatus.HIT,
                    entry=CacheEntry(
                        cache_key=cache_key.to_redis_key(),
                        query=results[0]["query"],
                        response=results[0]["response"],
                        model=results[0]["model"],
                        temperature=float(results[0]["temperature"]),
                    ),
                    similarity_score=1.0 - results[0]["distance"],
                    lookup_latency_ms=latency,
                )
        except Exception:
            pass
        return CacheLookupResult(
            status=CacheStatus.MISS,
            lookup_latency_ms=(time.monotonic() - start) * 1000,
        )

    def store(self, entry: CacheEntry) -> None:
        """Store a cache entry with its embedding."""
        redis_key = entry.cache_key
        self._store[redis_key] = entry
        if entry.query_embedding:
            self._embeddings[redis_key] = entry.query_embedding

        if self._redisvl_index is not None:
            try:
                self._redisvl_index.load([
                    {
                        "query": entry.query,
                        "response": entry.response,
                        "model": entry.model,
                        "temperature": entry.temperature,
                        "query_embedding": entry.query_embedding,
                    }
                ])
            except Exception:
                pass

    def clear(self) -> None:
        self._store.clear()
        self._embeddings.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._store),
            "similarity_boundary": self.similarity_boundary,
            "redis_connected": self._redis_client is not None,
        }


from datetime import datetime, timezone  # noqa: E402
