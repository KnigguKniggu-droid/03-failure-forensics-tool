"""OpenAI-compatible FastAPI proxy with semantic caching middleware."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.cache import SemanticCache, compute_cache_key
from src.models import CacheStatus, ProxyRequest, ProxyResponse
from src.streaming import StreamBuffer

app = FastAPI(
    title="Semantic Caching Proxy",
    description="OpenAI-compatible proxy with Redis VL semantic caching",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = SemanticCache()
_upstream_url = os.environ.get("UPSTREAM_URL", "https://api.openai.com/v1")
_api_key = os.environ.get("OPENAI_API_KEY", "")


def _extract_system_prompt(messages: list[dict[str, str]]) -> str:
    return next((m["content"] for m in messages if m.get("role") == "system"), "")


def _build_query_text(messages: list[dict[str, str]]) -> str:
    return " ".join(m["content"] for m in messages if m.get("role") != "system")


async def _get_embedding(text: str) -> list[float]:
    """Get embedding for cache lookup. Falls back to simple hash vector."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_upstream_url}/embeddings",
                json={"model": "text-embedding-3-small", "input": text},
                headers={"Authorization": f"Bearer {_api_key}"},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
    except Exception:
        return [float(hash(text) % 1000) / 1000.0] * 1536


@app.post("/v1/chat/completions")
async def chat_completions(request: ProxyRequest) -> Any:
    """OpenAI-compatible endpoint with semantic caching."""
    start = time.monotonic()
    system_prompt = _extract_system_prompt(request.messages)
    query_text = _build_query_text(request.messages)

    cache_key = compute_cache_key(
        model=request.model,
        temperature=request.temperature,
        system_prompt=system_prompt,
        messages=request.messages,
    )

    query_embedding = await _get_embedding(query_text)
    lookup = _cache.lookup(query_embedding, cache_key)

    if lookup.status == CacheStatus.HIT and lookup.entry:
        latency = (time.monotonic() - start) * 1000
        return ProxyResponse(
            id=str(uuid.uuid4()),
            model=request.model,
            content=lookup.entry.response,
            cache_status=CacheStatus.HIT,
            similarity_score=lookup.similarity_score,
            output_tokens=lookup.entry.output_tokens,
            latency_ms=latency,
            cached=True,
        )

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }
    headers = {"Authorization": f"Bearer {_api_key}", "Content-Type": "application/json"}

    if request.stream:
        async def stream_upstream() -> Any:
            buffer = StreamBuffer(_cache, cache_key)
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{_upstream_url}/chat/completions",
                    json=payload, headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_text():
                        yield chunk

        return StreamingResponse(stream_upstream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{_upstream_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = data.get("usage", {}).get("completion_tokens", 0)
    latency = (time.monotonic() - start) * 1000

    from src.models import CacheEntry
    entry = CacheEntry(
        cache_key=cache_key.to_redis_key(),
        query=query_text,
        query_embedding=query_embedding,
        response=content,
        model=request.model,
        temperature=request.temperature,
        system_prompt=system_prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    _cache.store(entry)

    return ProxyResponse(
        id=str(uuid.uuid4()),
        model=request.model,
        content=content,
        cache_status=CacheStatus.MISS,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency,
        cached=False,
    )


@app.get("/v1/cache/stats")
async def cache_stats() -> dict[str, Any]:
    return _cache.stats()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
