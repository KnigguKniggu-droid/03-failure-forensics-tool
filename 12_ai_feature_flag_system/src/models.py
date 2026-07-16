"""Typed contracts for the AI feature flag system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RolloutStrategy(str, Enum):
    PERCENTAGE = "percentage"
    CANARY = "canary"
    INSTANT = "instant"
    SCHEDULED = "scheduled"


class FlagStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ROLLED_BACK = "rolled_back"
    FULLY_ROLLED_OUT = "fully_rolled_out"


class QualityMetric(BaseModel):
    """A quality metric tracked by the LLM-as-judge evaluation window."""

    name: str
    baseline_score: float = Field(..., ge=0.0, le=1.0)
    canary_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(0.05, ge=0.0, le=1.0, description="Minimum acceptable delta")
    sample_count: int = 0
    is_passing: bool = True


class FeatureFlag(BaseModel):
    """A feature flag configuration with rollout rules."""

    flag_id: str = Field(..., description="Unique flag identifier")
    name: str
    description: str = ""
    enabled: bool = True
    status: FlagStatus = FlagStatus.ACTIVE
    rollout_strategy: RolloutStrategy = RolloutStrategy.CANARY
    rollout_percentage: float = Field(0.0, ge=0.0, le=1.0, description="Current rollout percentage")
    target_percentage: float = Field(1.0, ge=0.0, le=1.0, description="Final target percentage")
    incremental_step: float = Field(0.05, ge=0.01, le=0.5, description="Rollout increment per advancement")
    canary_metrics: list[QualityMetric] = Field(default_factory=list)
    error_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Error rate to trigger rollback")
    error_count_window: int = Field(100, ge=10, description="Sample window for error counting")
    auto_rollback: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlagEvaluation(BaseModel):
    """Result of evaluating a flag for a specific user."""

    flag_id: str
    user_id: str
    enabled: bool
    rollout_percentage: float
    evaluation_reason: str = Field("", description="Why the flag evaluated this way")
    hash_value: int = 0


class RolloutDecision(BaseModel):
    """Decision on whether to advance, hold, or rollback a rollout."""

    flag_id: str
    action: str = Field(..., description="advance | hold | rollback | complete")
    current_percentage: float
    new_percentage: float = Field(..., description="Proposed new percentage")
    reason: str = ""
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorSpike(BaseModel):
    """Recorded error spike triggering rollback."""

    flag_id: str
    error_count: int
    error_rate: float
    threshold: float
    window_size: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
