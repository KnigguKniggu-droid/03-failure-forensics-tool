"""Typed contracts for the arbitration system using instructor definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CriticType(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    LOGICAL_CONSISTENCY = "logical_consistency"
    COMPLETENESS = "completeness"


class CriticInput(BaseModel):
    """Input to a critic node."""

    query: str
    response: str
    context: str = Field("", description="Supporting context or reference material")
    critic_type: CriticType


class CriticOutput(BaseModel):
    """Typed output from a critic node using instructor definitions."""

    critic_type: CriticType
    score: float = Field(..., ge=0.0, le=10.0, description="Score from 0 to 10")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, description="Evidence supporting the score")
    critique: str = Field(..., description="Detailed critique text")
    model_used: str = ""
    latency_ms: float = 0.0


class AdjudicationInput(BaseModel):
    """Input to the central adjudicator node."""

    query: str
    response: str
    critic_outputs: list[CriticOutput]


class AdjudicationResult(BaseModel):
    """Final adjudication result from the central adjudicator."""

    final_score: int = Field(..., ge=1, le=10, description="Final score from 1 to 10")
    verdict: str = Field(..., description="accept | reject | revise")
    reasoning: str
    evidence_chain: list[str] = Field(default_factory=list, description="Chain of evidence from all critics")
    critic_scores: dict[str, float] = Field(default_factory=dict)
    score_deviations: dict[str, float] = Field(default_factory=dict, description="Deviation of each critic from final")
    consensus_level: float = Field(..., ge=0.0, le=1.0, description="How much critics agreed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArbitrationRequest(BaseModel):
    """API request for arbitration."""

    query: str
    response: str
    context: str = ""


class ArbitrationResponse(BaseModel):
    """API response containing the full arbitration result."""

    request_id: str
    adjudication: AdjudicationResult
    critic_outputs: list[CriticOutput]
    total_latency_ms: float
