"""Distributed Redis token bucket rate limiter.

Enforces atomic rate-limiting using distributed Redis token bucket keys
with Lua scripting for atomicity.
"""

from __future__ import annotations

import time
from typing import Any

from src.models import RateLimitResult

LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

local elapsed = now - last_refill
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
local retry_after = 0

if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
else
    retry_after = (requested - tokens) / refill_rate
end

redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, 300)

return {allowed, tostring(tokens), tostring(retry_after * 1000)}
"""


class TokenBucketRateLimiter:
    """Redis-backed token bucket rate limiter with atomic operations."""

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self.redis_url = redis_url
        self._redis: Any = None
        self._lua_sha: str | None = None
        self._local_buckets: dict[str, dict[str, float]] = {}

    def connect(self) -> None:
        """Connect to Redis and load the Lua script."""
        try:
            import redis
            self._redis = redis.from_url(self.redis_url)
            self._lua_sha = self._redis.script_load(LUA_TOKEN_BUCKET)
        except ImportError:
            self._redis = None

    def check_rate_limit(
        self,
        vendor_id: str,
        capacity: int,
        refill_rate: float,
        requested: int = 1,
    ) -> RateLimitResult:
        """Check if a request is allowed under the rate limit.

        Uses Redis Lua scripting for atomic token consumption when
        Redis is available, falls back to in-memory tracking otherwise.
        """
        key = f"ratelimit:{vendor_id}"
        now = time.time()

        if self._redis is not None and self._lua_sha is not None:
            try:
                result = self._redis.evalsha(
                    self._lua_sha, 1, key,
                    capacity, refill_rate, now, requested,
                )
                allowed = bool(int(result[0]))
                remaining = float(result[1])
                retry_after = float(result[2])
                return RateLimitResult(
                    allowed=allowed,
                    remaining_tokens=remaining,
                    retry_after_ms=retry_after,
                    limit=capacity,
                    burst=capacity,
                )
            except Exception:
                pass

        return self._local_check(key, capacity, refill_rate, now, requested)

    def _local_check(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
        now: float,
        requested: int,
    ) -> RateLimitResult:
        """In-memory fallback rate limiting."""
        bucket = self._local_buckets.get(key, {"tokens": float(capacity), "last_refill": now})
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(float(capacity), bucket["tokens"] + elapsed * refill_rate)

        if bucket["tokens"] >= requested:
            bucket["tokens"] -= requested
            allowed = True
            retry_after = 0.0
        else:
            allowed = False
            retry_after = (requested - bucket["tokens"]) / refill_rate * 1000

        bucket["last_refill"] = now
        self._local_buckets[key] = bucket

        return RateLimitResult(
            allowed=allowed,
            remaining_tokens=bucket["tokens"],
            retry_after_ms=retry_after,
            limit=capacity,
            burst=capacity,
        )
