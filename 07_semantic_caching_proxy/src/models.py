"""Typed contracts for the semantic caching proxy."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CacheStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    PARTIAL_HIT = "partial_hit"


class CacheKey(BaseModel):
    """Composite cache key embedding system prompt, temperature, and model metadata."""

    system_hash: str = Field(..., description="Hash of the system prompt")
    temperature_hash: str = Field(..., description="Hash of the temperature setting")
    model_hash: str = Field(..., description="Hash of the model identifier")
    combined_hash: str = Field(..., description="SHA-256 of all component hashes")
    model: str
    temperature: float
    system_prompt: str = ""

    def to_redis_key(self) -> str:
        return f"semcache:{self.combined_hash}"


class CacheEntry(BaseModel):
    """A cached response entry stored in Redis VL."""

    cache_key: str
    query: str
    query_embedding: list[float] = Field(default_factory=list)
    response: str
    response_embedding: list[float] = Field(default_factory=list)
    model: str
    temperature: float
    system_prompt: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600


class CacheLookupResult(BaseModel):
    """Result of a semantic cache lookup."""

    status: CacheStatus
    entry: CacheEntry | None = None
    similarity_score: float = Field(0.0, ge=0.0, le=1.0)
    lookup_latency_ms: float = 0.0


class ProxyRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class ProxyResponse(BaseModel):
    """Response from the proxy, either from cache or upstream."""

    id: str
    model: str
    content: str
    cache_status: CacheStatus
    similarity_score: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
