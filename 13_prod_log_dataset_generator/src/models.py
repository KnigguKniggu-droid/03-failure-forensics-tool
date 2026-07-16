"""Typed contracts for the production log dataset generator."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LogSource(str, Enum):
    CLICKHOUSE = "clickhouse"
    POSTGRES = "postgres"
    FILE = "file"


class ProductionLog(BaseModel):
    """A single production LLM interaction log entry."""

    log_id: str
    timestamp: datetime
    user_query: str = Field(..., min_length=1)
    model_response: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None
    user_feedback: str | None = Field(None, description="thumbs up/down or text feedback")
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class Cluster(BaseModel):
    """A cluster of similar production logs identified by HDBSCAN."""

    cluster_id: int = Field(..., description="HDBSCAN cluster label (-1 = noise)")
    label: str = Field("", description="Human-readable cluster label")
    log_ids: list[str] = Field(default_factory=list)
    centroid_embedding: list[float] | None = None
    size: int = 0
    coherence_score: float = Field(0.0, ge=0.0, le=1.0, description="Intra-cluster semantic coherence")
    representative_query: str = Field("", description="Most representative query in cluster")
    keywords: list[str] = Field(default_factory=list)


class EvalDatasetItem(BaseModel):
    """A single evaluation dataset item generated from production logs."""

    item_id: str
    query: str
    expected_response: str = ""
    cluster_id: int
    difficulty: str = Field("medium", description="easy | medium | hard")
    source_log_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A generated evaluation dataset from production logs."""

    dataset_id: str
    name: str
    description: str = ""
    items: list[EvalDatasetItem] = Field(default_factory=list)
    source: LogSource
    total_logs_mined: int = 0
    total_clusters: int = 0
    noise_points: int = 0
    coverage_score: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of non-noise logs covered")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    clustering_params: dict[str, Any] = Field(default_factory=dict)


class MiningConfig(BaseModel):
    """Configuration for production log mining."""

    source: LogSource = LogSource.CLICKHOUSE
    connection_url: str = ""
    query: str = Field("", description="SQL query to fetch logs")
    time_range_hours: int = Field(168, ge=1, description="Lookback window in hours")
    min_query_length: int = Field(10, ge=1)
    max_logs: int = Field(10000, ge=100)
    min_cluster_size: int = Field(5, ge=2, description="HDBSCAN min_cluster_size")
    min_samples: int = Field(3, ge=1, description="HDBSCAN min_samples")
    items_per_cluster: int = Field(3, ge=1, description="Max items to sample per cluster")
    embedding_model: str = "text-embedding-3-small"
