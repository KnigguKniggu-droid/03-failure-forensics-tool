"""Streaming response handler that buffers finalized output into Redis.

Streams partial data responses to users while simultaneously buffering
the complete stream output for cache storage.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

from src.models import CacheEntry, CacheKey
from src.cache import SemanticCache


class StreamBuffer:
    """Buffers a streaming response for cache storage while yielding chunks to the user."""

    def __init__(self, cache: SemanticCache, cache_key: CacheKey) -> None:
        self.cache = cache
        self.cache_key = cache_key
        self._buffer: list[str] = []
        self._full_response: str = ""

    async def buffer_and_stream(
        self,
        upstream_stream: AsyncGenerator[str, None],
        query: str,
        query_embedding: list[float],
        model: str,
    ) -> AsyncGenerator[str, None]:
        """Yield chunks to the user while buffering the full response."""
        chunks: list[str] = []

        async for chunk in upstream_stream:
            chunks.append(chunk)
            self._buffer.append(chunk)
            yield chunk

        self._full_response = "".join(chunks)
        await self._store_to_cache(query, query_embedding, model)

    async def _store_to_cache(self, query: str, query_embedding: list[float], model: str) -> None:
        """Store the buffered response into the semantic cache."""
        if not self._full_response:
            return
        entry = CacheEntry(
            cache_key=self.cache_key.to_redis_key(),
            query=query,
            query_embedding=query_embedding,
            response=self._full_response,
            model=model,
            temperature=self.cache_key.temperature,
            system_prompt=self.cache_key.system_prompt,
            output_tokens=max(1, len(self._full_response) // 4),
        )
        await asyncio.get_event_loop().run_in_executor(None, self.cache.store, entry)

    @property
    def buffered_response(self) -> str:
        return self._full_response
