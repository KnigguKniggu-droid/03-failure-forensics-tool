"""Typed contracts for the LLM API gateway."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class VendorConfig(BaseModel):
    """Configuration for an upstream LLM vendor."""

    vendor_id: str
    display_name: str
    api_base: str
    api_key_env: str
    model_mapping: dict[str, str] = Field(default_factory=dict)
    priority: int = Field(..., ge=0, description="Lower = higher priority")
    rate_limit_rps: int = Field(..., ge=1, description="Requests per second")
    rate_limit_burst: int = Field(..., ge=1, description="Burst capacity")
    timeout_ms: int = Field(30000, ge=1000)
    circuit_failure_threshold: int = Field(5, ge=1)
    circuit_recovery_timeout_s: int = Field(30, ge=1)
    latency_warning_ms: float = Field(2000, ge=100)
    latency_critical_ms: float = Field(5000, ge=500)


class GatewayConfig(BaseModel):
    """Top-level gateway configuration."""

    vendors: list[VendorConfig] = Field(..., min_length=1)
    redis_url: str = "redis://localhost:6379"
    enable_observability: bool = True
    enable_rate_limiting: bool = True
    enable_fallback: bool = True
    default_vendor: str = ""


class RateLimitResult(BaseModel):
    """Result of a rate limit check."""

    allowed: bool
    remaining_tokens: float = 0.0
    retry_after_ms: float = 0.0
    limit: int = 0
    burst: int = 0


class CircuitBreakerState(BaseModel):
    """Current state of a vendor circuit breaker."""

    vendor_id: str
    state: CircuitState
    failure_count: int = 0
    failure_threshold: int = 5
    last_failure_time: datetime | None = None
    recovery_started_at: datetime | None = None
    consecutive_successes: int = 0


class GatewayRequest(BaseModel):
    """Incoming request to the gateway."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False
    preferred_vendor: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayResponse(BaseModel):
    """Response from the gateway after routing and execution."""

    request_id: str
    vendor_id: str
    model: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    rate_limited: bool = False
    fallback_used: bool = False
    circuit_state: CircuitState = CircuitState.CLOSED
    error: str | None = None


class LatencyMetric(BaseModel):
    """Rolling window latency metric for a vendor."""

    vendor_id: str
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    sample_count: int = 0
    window_seconds: int = 60
