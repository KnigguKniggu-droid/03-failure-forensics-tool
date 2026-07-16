"""Typed contracts for the realtime LLM observability system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    LATENCY = "latency"
    TOKEN_COUNT = "token_count"
    TOKEN_DRIFT = "token_drift"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    STREAM_TTFB = "stream_ttfb"  # Time to first byte


class StreamingMetric(BaseModel):
    """A single real-time metric emitted during LLM inference streaming."""

    metric_id: str
    request_id: str
    model: str
    metric_type: MetricType
    value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    stream_chunk_index: int = Field(-1, description="Chunk index in streaming response, -1 for non-streaming")


class LatencyBreakdown(BaseModel):
    """Detailed latency breakdown for a single request."""

    request_id: str
    model: str
    ttfb_ms: float = Field(0.0, ge=0.0, description="Time to first byte")
    total_latency_ms: float = Field(0.0, ge=0.0)
    streaming_duration_ms: float = Field(0.0, ge=0.0)
    prefill_latency_ms: float = Field(0.0, ge=0.0, description="Time before first token")
    decode_latency_ms: float = Field(0.0, ge=0.0, description="Time for token generation")
    chunks_received: int = 0
    avg_inter_chunk_ms: float = Field(0.0, ge=0.0)


class TokenDriftMetric(BaseModel):
    """Token count drift metric comparing expected vs actual token counts."""

    request_id: str
    model: str
    expected_tokens: int
    actual_tokens: int
    drift_ratio: float = Field(..., description="actual / expected (1.0 = no drift)")
    drift_direction: str = Field("", description="over | under | stable")
    is_anomaly: bool = False
    threshold: float = Field(0.2, description="Drift ratio threshold for anomaly flagging")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RollingWindowMetric(BaseModel):
    """Aggregated metric over a rolling time window."""

    metric_type: MetricType
    model: str
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    sample_count: int = 0
    window_seconds: int = 60
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DashboardSnapshot(BaseModel):
    """Complete dashboard snapshot with all current metrics."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_metrics: dict[str, RollingWindowMetric] = Field(default_factory=dict)
    token_drift_metrics: dict[str, RollingWindowMetric] = Field(default_factory=dict)
    throughput_metrics: dict[str, RollingWindowMetric] = Field(default_factory=dict)
    active_streams: int = 0
    total_requests_window: int = 0
    error_rate: float = 0.0
    anomalies: list[TokenDriftMetric] = Field(default_factory=list)
    model_health: dict[str, dict[str, Any]] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    """A streaming event from an LLM inference call."""

    request_id: str
    model: str
    event_type: str = Field(..., description="start | chunk | end | error")
    chunk_index: int = -1
    token_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    content: str = ""
    error: str | None = None
