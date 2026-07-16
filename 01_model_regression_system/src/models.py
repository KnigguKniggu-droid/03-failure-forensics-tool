"""Typed data contracts for the regression system.

All Pydantic models used across the regression pipeline live here so that
every module imports from a single source of truth for schema definitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EmailCategory(str, Enum):
    """Supported classification categories for customer support emails."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ExpectedDifficulty(str, Enum):
    """Difficulty tag assigned to each ground-truth test item."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PromptConfig(BaseModel):
    """Configuration contract loaded from YAML prompt files.

    Tracks version strings so that regression runs can be pinned to a
    specific prompt revision and compared across historical baselines.
    """

    prompt_id: str = Field(..., min_length=1, description="Unique prompt identifier")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version string")
    model: str = Field(..., description="Target model name (e.g. gpt-4o-mini)")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(50, ge=1, le=4096)
    system_prompt: str = Field(..., min_length=1)
    user_template: str = Field(..., min_length=1)
    categories: list[EmailCategory] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("categories")
    @classmethod
    def categories_must_include_all(cls, v: list[EmailCategory]) -> list[EmailCategory]:
        required = {EmailCategory.BILLING, EmailCategory.TECHNICAL, EmailCategory.ACCOUNT, EmailCategory.GENERAL}
        missing = required - set(v)
        if missing:
            raise ValueError(f"Missing required categories: {missing}")
        return v


class GroundTruthItem(BaseModel):
    """A single labeled test case used for regression scoring."""

    id: str = Field(..., description="Unique test item identifier")
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    expected_category: EmailCategory
    expected_difficulty: ExpectedDifficulty
    tags: list[str] = Field(default_factory=list)


class ModelPrediction(BaseModel):
    """Raw output from a single LLM classification call."""

    item_id: str
    predicted_category: EmailCategory | None = None
    raw_output: str = ""
    latency_ms: float = 0.0
    token_count: int = 0
    error: str | None = None


class ScoringResult(BaseModel):
    """Per-item scoring outcome combining accuracy and judge relevance."""

    item_id: str
    expected_category: EmailCategory
    predicted_category: EmailCategory | None
    is_correct: bool
    judge_relevance_score: float = Field(..., ge=0.0, le=1.0, description="LLM-as-judge relevance score")
    judge_explanation: str = ""
    difficulty: ExpectedDifficulty


class RegressionSeverity(str, Enum):
    """Severity level for regression detection thresholds."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


class RegressionReport(BaseModel):
    """Aggregate regression report comparing current run against baseline."""

    run_id: str
    prompt_id: str
    prompt_version: str
    model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_items: int
    correct_count: int
    accuracy: float = Field(..., ge=0.0, le=1.0)
    baseline_accuracy: float = Field(..., ge=0.0, le=1.0)
    regression_delta: float = Field(..., description="Current accuracy minus baseline accuracy (negative = regression)")
    severity: RegressionSeverity
    mean_judge_relevance: float = Field(..., ge=0.0, le=1.0)
    per_difficulty: dict[str, dict[str, float]] = Field(default_factory=dict)
    failed_items: list[ScoringResult] = Field(default_factory=list)
    statistical_significance: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.severity == RegressionSeverity.NONE

    @property
    def blocks_merge(self) -> bool:
        return self.severity == RegressionSeverity.CRITICAL
