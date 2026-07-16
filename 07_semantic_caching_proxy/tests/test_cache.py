"""Tests for the semantic cache key computation and lookup logic."""

from __future__ import annotations

import pytest

from src.cache import SemanticCache, compute_cache_key, cosine_similarity
from src.models import CacheEntry, CacheStatus


def test_cache_key_is_deterministic():
    key1 = compute_cache_key("gpt-4o", 0.7, "You are helpful", [{"role": "user", "content": "hi"}])
    key2 = compute_cache_key("gpt-4o", 0.7, "You are helpful", [{"role": "user", "content": "hi"}])
    assert key1.combined_hash == key2.combined_hash


def test_cache_key_differs_by_model():
    key1 = compute_cache_key("gpt-4o", 0.7, "system", [{"role": "user", "content": "hi"}])
    key2 = compute_cache_key("gpt-4o-mini", 0.7, "system", [{"role": "user", "content": "hi"}])
    assert key1.combined_hash != key2.combined_hash


def test_cache_key_differs_by_temperature():
    key1 = compute_cache_key("gpt-4o", 0.7, "system", [{"role": "user", "content": "hi"}])
    key2 = compute_cache_key("gpt-4o", 0.0, "system", [{"role": "user", "content": "hi"}])
    assert key1.combined_hash != key2.combined_hash


def test_cache_key_differs_by_system_prompt():
    key1 = compute_cache_key("gpt-4o", 0.7, "system A", [{"role": "user", "content": "hi"}])
    key2 = compute_cache_key("gpt-4o", 0.7, "system B", [{"role": "user", "content": "hi"}])
    assert key1.combined_hash != key2.combined_hash


def test_semantic_cache_hit_above_boundary():
    cache = SemanticCache(similarity_boundary=0.95)
    key = compute_cache_key("gpt-4o", 0.7, "sys", [{"role": "user", "content": "hi"}])
    entry = CacheEntry(
        cache_key=key.to_redis_key(),
        query="hello world",
        query_embedding=[1.0, 0.0, 0.0],
        response="cached response",
        model="gpt-4o",
        temperature=0.7,
    )
    cache.store(entry)
    result = cache.lookup([1.0, 0.0, 0.0], key)
    assert result.status == CacheStatus.HIT
    assert result.entry is not None
    assert result.entry.response == "cached response"


def test_semantic_cache_miss_below_boundary():
    cache = SemanticCache(similarity_boundary=0.95)
    key = compute_cache_key("gpt-4o", 0.7, "sys", [{"role": "user", "content": "hi"}])
    entry = CacheEntry(
        cache_key=key.to_redis_key(),
        query="hello world",
        query_embedding=[1.0, 0.0, 0.0],
        response="cached response",
        model="gpt-4o",
        temperature=0.7,
    )
    cache.store(entry)
    result = cache.lookup([0.0, 1.0, 0.0], key)
    assert result.status == CacheStatus.MISS


def test_cosine_similarity_identical():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cache_stats_returns_counts():
    cache = SemanticCache()
    key = compute_cache_key("gpt-4o", 0.7, "sys", [{"role": "user", "content": "hi"}])
    cache.store(CacheEntry(
        cache_key=key.to_redis_key(),
        query="test",
        query_embedding=[1.0],
        response="resp",
        model="gpt-4o",
        temperature=0.7,
    ))
    stats = cache.stats()
    assert stats["total_entries"] == 1
    assert stats["similarity_boundary"] == 0.95
