"""Typed contracts for the cost autopilot routing system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ComplexityTier(int, Enum):
    """Three strict complexity tiers for request routing."""

    TIER_1_EXTRACTION = 1  # Simple extraction: summarization, keyword extraction, entity pull
    TIER_2_CLASSIFICATION = 2  # Classification: sentiment, categorization, routing decisions
    TIER_3_MULTI_STEP_LOGIC = 3  # Multi-step logic: reasoning, code generation, analysis


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class ModelConfig(BaseModel):
    """Per-model configuration with cost and capability metadata."""

    model_id: str = Field(..., description="Model identifier (e.g. gpt-4o)")
    provider: ProviderType
    display_name: str
    input_cost_per_1k: float = Field(..., ge=0.0, description="USD per 1K input tokens")
    output_cost_per_1k: float = Field(..., ge=0.0, description="USD per 1K output tokens")
    max_context: int = Field(..., gt=0, description="Maximum context window in tokens")
    max_output: int = Field(..., gt=0)
    latency_p50_ms: float = Field(..., ge=0.0, description="Median latency in ms")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Normalized quality benchmark")
    supported_tiers: list[ComplexityTier] = Field(..., min_length=1)
    api_base: str = ""
    api_key_env: str = ""

    @property
    def cost_efficiency(self) -> float:
        """Quality per dollar ratio (higher is better)."""
        avg_cost = (self.input_cost_per_1k + self.output_cost_per_1k) / 2.0
        if avg_cost == 0:
            return self.quality_score * 1000.0
        return self.quality_score / avg_cost


class ComplexityClassification(BaseModel):
    """Result of classifying a request into a complexity tier."""

    tier: ComplexityTier
    confidence: float = Field(..., ge=0.0, le=1.0)
    signals: dict[str, Any] = Field(default_factory=dict)
    estimated_input_tokens: int = Field(..., ge=0)
    estimated_output_tokens: int = Field(..., ge=0)


class RoutingDecision(BaseModel):
    """The routing decision made by the proxy for a given request."""

    request_id: str
    selected_model: str
    selected_provider: ProviderType
    tier: ComplexityTier
    estimated_cost: float = Field(..., ge=0.0)
    estimated_latency_ms: float
    routing_reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RoutingFailure(BaseModel):
    """Recorded failure when a routed model produces poor output."""

    request_id: str
    model_id: str
    tier: ComplexityTier
    failure_type: str = Field(..., description="quality | latency | error | cost_overrun")
    expected_quality: float
    actual_quality: float
    actual_cost: float
    actual_latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_signal: float = Field(..., description="Negative penalty signal for model adaptation")


class AdaptationMetrics(BaseModel):
    """Weekly aggregate metrics for model adaptation."""

    week_start: datetime
    model_id: str
    total_requests: int
    failure_count: int
    failure_rate: float = Field(..., ge=0.0, le=1.0)
    avg_quality: float = Field(..., ge=0.0, le=1.0)
    avg_cost: float
    avg_latency_ms: float
    adjusted_quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score after adaptation penalty")
    total_spend: float
    recommended_action: str = Field(..., description="keep | demote | promote | investigate")


class ProxyRequest(BaseModel):
    """Incoming request to the proxy."""

    messages: list[dict[str, str]]
    max_tokens: int = Field(1000, ge=1, le=16000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProxyResponse(BaseModel):
    """Response from the proxy after routing and execution."""

    request_id: str
    model: str
    provider: ProviderType
    content: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    routing: RoutingDecision
    verification_score: float | None = None
