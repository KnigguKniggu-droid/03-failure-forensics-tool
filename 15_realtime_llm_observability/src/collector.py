"""Streaming metric collector for real-time LLM observability.

Collects metrics from LLM inference streams in real time, tracks per-request
latency breakdowns, and feeds data into the rolling window aggregator.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from typing import Any, AsyncGenerator

from src.models import (
    LatencyBreakdown,
    MetricType,
    StreamingMetric,
    StreamEvent,
    TokenDriftMetric,
)

MAX_BUFFER_SIZE = 10000


class MetricCollector:
    """Collects and buffers real-time streaming metrics."""

    def __init__(self, window_seconds: int = 60) -> None:
        self.window_seconds = window_seconds
        self._metric_buffer: deque[StreamingMetric] = deque(maxlen=MAX_BUFFER_SIZE)
        self._stream_timings: dict[str, dict[str, float]] = defaultdict(dict)
        self._stream_chunks: dict[str, list[float]] = defaultdict(list)
        self._active_streams: set[str] = set()
        self._lock = asyncio.Lock()

    async def record_stream_start(self, request_id: str, model: str) -> None:
        """Record the start of a streaming inference call."""
        async with self._lock:
            self._stream_timings[request_id] = {
                "start": time.monotonic(),
                "model": 0.0,
                "first_chunk": 0.0,
                "end": 0.0,
            }
            self._stream_chunks[request_id] = []
            self._active_streams.add(request_id)

        await self._emit_metric(
            request_id=request_id,
            model=model,
            metric_type=MetricType.STREAM_TTFB,
            value=0.0,
            stream_chunk_index=-1,
        )

    async def record_stream_chunk(
        self,
        request_id: str,
        model: str,
        chunk_index: int,
        token_count: int = 0,
    ) -> None:
        """Record a chunk received during streaming."""
        now = time.monotonic()
        async with self._lock:
            timings = self._stream_timings.get(request_id, {})
            if "first_chunk" not in timings or timings.get("first_chunk", 0) == 0:
                timings["first_chunk"] = now
                start = timings.get("start", now)
                ttfb = (now - start) * 1000

                await self._emit_metric(
                    request_id=request_id,
                    model=model,
                    metric_type=MetricType.STREAM_TTFB,
                    value=ttfb,
                    stream_chunk_index=chunk_index,
                )

            self._stream_chunks[request_id].append(now)

            if len(self._stream_chunks[request_id]) > 1:
                prev = self._stream_chunks[request_id][-2]
                inter_chunk = (now - prev) * 1000
                await self._emit_metric(
                    request_id=request_id,
                    model=model,
                    metric_type=MetricType.LATENCY,
                    value=inter_chunk,
                    stream_chunk_index=chunk_index,
                    metadata={"type": "inter_chunk"},
                )

            await self._emit_metric(
                request_id=request_id,
                model=model,
                metric_type=MetricType.TOKEN_COUNT,
                value=float(token_count),
                stream_chunk_index=chunk_index,
            )

    async def record_stream_end(
        self,
        request_id: str,
        model: str,
        total_tokens: int,
        expected_tokens: int = 0,
    ) -> None:
        """Record the end of a streaming inference call."""
        now = time.monotonic()
        async with self._lock:
            timings = self._stream_timings.get(request_id, {})
            timings["end"] = now
            start = timings.get("start", now)
            total_latency = (now - start) * 1000
            self._active_streams.discard(request_id)

        await self._emit_metric(
            request_id=request_id,
            model=model,
            metric_type=MetricType.LATENCY,
            value=total_latency,
            stream_chunk_index=-1,
            metadata={"type": "total"},
        )

        await self._emit_metric(
            request_id=request_id,
            model=model,
            metric_type=MetricType.TOKEN_COUNT,
            value=float(total_tokens),
            stream_chunk_index=-1,
            metadata={"type": "total"},
        )

        if expected_tokens > 0:
            drift = total_tokens / expected_tokens if expected_tokens > 0 else 1.0
            await self._emit_metric(
                request_id=request_id,
                model=model,
                metric_type=MetricType.TOKEN_DRIFT,
                value=drift,
                stream_chunk_index=-1,
                metadata={
                    "expected": expected_tokens,
                    "actual": total_tokens,
                    "is_anomaly": abs(drift - 1.0) > 0.2,
                },
            )

    async def record_error(self, request_id: str, model: str, error: str) -> None:
        """Record a streaming error."""
        async with self._lock:
            self._active_streams.discard(request_id)

        await self._emit_metric(
            request_id=request_id,
            model=model,
            metric_type=MetricType.ERROR_RATE,
            value=1.0,
            stream_chunk_index=-1,
            metadata={"error": error},
        )

    async def _emit_metric(
        self,
        request_id: str,
        model: str,
        metric_type: MetricType,
        value: float,
        stream_chunk_index: int = -1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metric = StreamingMetric(
            metric_id=str(uuid.uuid4()),
            request_id=request_id,
            model=model,
            metric_type=metric_type,
            value=value,
            stream_chunk_index=stream_chunk_index,
            metadata=metadata or {},
        )
        self._metric_buffer.append(metric)

    def get_metrics(
        self,
        metric_type: MetricType | None = None,
        model: str | None = None,
        window_seconds: int | None = None,
    ) -> list[StreamingMetric]:
        """Get buffered metrics, optionally filtered by type and model."""
        window = window_seconds or self.window_seconds
        cutoff = time.time() - window
        results: list[StreamingMetric] = []
        for m in self._metric_buffer:
            if m.timestamp.timestamp() < cutoff:
                continue
            if metric_type and m.metric_type != metric_type:
                continue
            if model and m.model != model:
                continue
            results.append(m)
        return results

    @property
    def active_stream_count(self) -> int:
        return len(self._active_streams)

    def get_latency_breakdown(self, request_id: str) -> LatencyBreakdown | None:
        """Get a detailed latency breakdown for a specific request."""
        timings = self._stream_timings.get(request_id)
        chunks = self._stream_chunks.get(request_id, [])
        if not timings:
            return None

        start = timings.get("start", 0)
        first = timings.get("first_chunk", 0)
        end = timings.get("end", 0)

        ttfb = (first - start) * 1000 if first and start else 0.0
        total = (end - start) * 1000 if end and start else 0.0
        streaming = (end - first) * 1000 if end and first else 0.0

        inter_chunk_times: list[float] = []
        for i in range(1, len(chunks)):
            inter_chunk_times.append((chunks[i] - chunks[i - 1]) * 1000)

        avg_inter = sum(inter_chunk_times) / len(inter_chunk_times) if inter_chunk_times else 0.0

        model = "unknown"
        for m in self._metric_buffer:
            if m.request_id == request_id:
                model = m.model
                break

        return LatencyBreakdown(
            request_id=request_id,
            model=model,
            ttfb_ms=ttfb,
            total_latency_ms=total,
            streaming_duration_ms=streaming,
            prefill_latency_ms=ttfb,
            decode_latency_ms=streaming,
            chunks_received=len(chunks),
            avg_inter_chunk_ms=avg_inter,
        )
