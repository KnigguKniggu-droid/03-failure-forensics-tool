"""Typed contracts for pipeline spans and forensic analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineStep(str, Enum):
    """The four canonical pipeline steps."""

    INTAKE = "intake"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    DEGRADED = "degraded"


class SpanData(BaseModel):
    """Standard span object capturing pipeline step metrics.

    Every step injects context metrics including latency, token sizes,
    and an explicit model self-confidence score from 1 to 5.
    """

    span_id: str
    trace_id: str
    step: PipelineStep
    status: SpanStatus = SpanStatus.OK
    latency_ms: float = Field(..., ge=0.0)
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    model_confidence: int = Field(..., ge=1, le=5, description="Model self-confidence score 1-5")
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_span_id: str | None = None


class TraceRecord(BaseModel):
    """A complete trace containing all spans for a single pipeline run."""

    trace_id: str
    spans: list[SpanData] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    final_output: str = ""
    final_status: SpanStatus = SpanStatus.OK

    def add_span(self, span: SpanData) -> None:
        self.spans.append(span)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    @property
    def min_confidence(self) -> int:
        if not self.spans:
            return 5
        return min(s.model_confidence for s in self.spans)


class FaultTaxonomy(str, Enum):
    """Automated taxonomy for classifying pipeline failures."""

    EXTRACTION_HALLUCINATION = "extraction_hallucination"
    MISCLASSIFICATION = "misclassification"
    PROPAGATION_ERROR = "propagation_error"
    PROMPT_FAILURE = "prompt_failure"
    CONTEXT_LOSS = "context_loss"


class ForensicFinding(BaseModel):
    """A single forensic finding from backward root cause analysis."""

    finding_id: str
    trace_id: str
    fault_type: FaultTaxonomy
    root_cause_span: PipelineStep
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    remediation: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ForensicReport(BaseModel):
    """Complete forensic report for a trace."""

    trace_id: str
    findings: list[ForensicFinding] = Field(default_factory=list)
    overall_assessment: str = ""
    severity: str = Field("low", description="low | medium | high | critical")
    total_latency_ms: float = 0.0
    min_confidence: int = 5
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineInput(BaseModel):
    """Input to the 4-step pipeline."""

    raw_text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineOutput(BaseModel):
    """Final output from the 4-step pipeline."""

    trace_id: str
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    classification: str = ""
    summary: str = ""
    confidence_scores: dict[str, int] = Field(default_factory=dict)
    spans: list[SpanData] = Field(default_factory=list)
