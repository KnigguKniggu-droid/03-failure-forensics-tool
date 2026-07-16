"""Typed contracts for the prompt A/B testing platform."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    KILLED = "killed"


class VariantAllocation(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


class PromptVersion(BaseModel):
    """A versioned prompt configuration in the registry."""

    prompt_id: str = Field(..., description="Unique prompt identifier")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    system_prompt: str
    user_template: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    changelog: str = Field("", description="Developer change message")
    created_by: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class ExperimentVariant(BaseModel):
    """A variant in an A/B experiment."""

    variant_id: str
    allocation: VariantAllocation
    prompt_version: PromptVersion
    traffic_percentage: float = Field(..., ge=0.0, le=1.0)


class Experiment(BaseModel):
    """An A/B test experiment comparing prompt variants."""

    experiment_id: str
    name: str
    description: str = ""
    variants: list[ExperimentVariant] = Field(..., min_length=2)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    started_at: datetime | None = None
    ended_at: datetime | None = None
    kill_switch_threshold: float = Field(0.05, description="Minimum performance delta to trigger kill switch")
    statistical_significance: float = Field(0.05, description="p-value threshold for significance")
    sample_size_target: int = Field(1000, ge=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentOutcome(BaseModel):
    """A single outcome observation from an experiment participant."""

    experiment_id: str
    variant_id: str
    user_id: str
    success: bool
    score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatisticalResult(BaseModel):
    """Result of statistical analysis on experiment outcomes."""

    experiment_id: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    winner: str = Field("", description="control | treatment | inconclusive")
    effect_size: float = Field(..., description="Cohen's d")


class KillSwitchResult(BaseModel):
    """Result of kill switch evaluation."""

    experiment_id: str
    triggered: bool
    reason: str = ""
    performance_delta: float
    threshold: float
    variant_killed: str = ""


class TrafficAssignment(BaseModel):
    """Traffic assignment for a user session."""

    user_id: str
    experiment_id: str
    variant_id: str
    allocation: VariantAllocation
    hash_value: int
