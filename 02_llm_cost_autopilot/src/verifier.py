"""Downstream verification evaluator.

Samples routed responses and uses a judge model to verify output quality.
Feeds routing failures back into the adaptation metrics pipeline so that
the model registry can be adjusted weekly.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from src.models import (
    AdaptationMetrics,
    ComplexityTier,
    ModelConfig,
    ProxyResponse,
    RoutingFailure,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "autopilot.db"

VERIFICATION_PROMPT = """You are a quality evaluator. Score the following LLM response
on a scale of 0.0 to 1.0 for correctness, completeness, and coherence.

Request messages: {request}
Response: {response}

Respond in JSON: {{"quality_score": float, "reason": "brief explanation"}}"""


class VerificationResult(BaseModel, extra="forbid"):  # type: ignore
    quality_score: float
    reason: str


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS routing_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            tier INTEGER NOT NULL,
            failure_type TEXT NOT NULL,
            expected_quality REAL NOT NULL,
            actual_quality REAL NOT NULL,
            actual_cost REAL NOT NULL,
            actual_latency_ms REAL NOT NULL,
            feedback_signal REAL NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS adaptation_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start TEXT NOT NULL,
            model_id TEXT NOT NULL,
            total_requests INTEGER NOT NULL,
            failure_count INTEGER NOT NULL,
            failure_rate REAL NOT NULL,
            avg_quality REAL NOT NULL,
            avg_cost REAL NOT NULL,
            avg_latency_ms REAL NOT NULL,
            adjusted_quality_score REAL NOT NULL,
            total_spend REAL NOT NULL,
            recommended_action TEXT NOT NULL,
            UNIQUE(week_start, model_id)
        );

        CREATE INDEX IF NOT EXISTS idx_failures_model ON routing_failures(model_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_week ON adaptation_metrics(week_start, model_id);
        """
    )
    conn.commit()


def get_db_conn(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def should_verify(sample_rate: float = 0.1) -> bool:
    return random.random() < sample_rate


async def verify_response(
    response: ProxyResponse,
    request_messages: list[dict[str, str]],
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
) -> float:
    """Use a judge model to score response quality. Returns 0.0 to 1.0."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    prompt = VERIFICATION_PROMPT.format(
        request=json.dumps(request_messages[:3], ensure_ascii=False)[:2000],
        response=response.content[:2000],
    )
    payload = {
        "model": judge_model,
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a strict quality evaluator."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            return max(0.0, min(1.0, float(parsed.get("quality_score", 0.0))))
    except Exception:
        return 0.0


def record_failure(
    conn: sqlite3.Connection,
    response: ProxyResponse,
    model_config: ModelConfig,
    actual_quality: float,
    failure_type: str = "quality",
) -> RoutingFailure:
    expected = model_config.quality_score
    feedback = actual_quality - expected

    failure = RoutingFailure(
        request_id=response.request_id,
        model_id=response.model,
        tier=response.routing.tier,
        failure_type=failure_type,
        expected_quality=expected,
        actual_quality=actual_quality,
        actual_cost=response.cost,
        actual_latency_ms=response.latency_ms,
        feedback_signal=feedback,
    )

    conn.execute(
        """
        INSERT INTO routing_failures
            (request_id, model_id, tier, failure_type, expected_quality,
             actual_quality, actual_cost, actual_latency_ms, feedback_signal, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            failure.request_id,
            failure.model_id,
            failure.tier.value,
            failure.failure_type,
            failure.expected_quality,
            failure.actual_quality,
            failure.actual_cost,
            failure.actual_latency_ms,
            failure.feedback_signal,
            failure.timestamp.isoformat(),
        ),
    )
    conn.commit()
    return failure


def compute_weekly_metrics(
    conn: sqlite3.Connection,
    model_id: str,
    week_start: datetime | None = None,
) -> AdaptationMetrics:
    """Aggregate routing failures into weekly adaptation metrics."""
    if week_start is None:
        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    week_end = week_start + timedelta(days=7)
    week_str = week_start.isoformat()

    rows = conn.execute(
        """
        SELECT * FROM routing_failures
        WHERE model_id = ? AND timestamp >= ? AND timestamp < ?
        """,
        (model_id, week_start.isoformat(), week_end.isoformat()),
    ).fetchall()

    total_failures = len(rows)
    total_requests = max(total_failures * 10, 1)
    failure_rate = total_failures / total_requests if total_requests > 0 else 0.0

    if total_failures > 0:
        avg_quality = sum(r["actual_quality"] for r in rows) / total_failures
        avg_cost = sum(r["actual_cost"] for r in rows) / total_failures
        avg_latency = sum(r["actual_latency_ms"] for r in rows) / total_failures
        total_spend = sum(r["actual_cost"] for r in rows)
        avg_feedback = sum(r["feedback_signal"] for r in rows) / total_failures
    else:
        avg_quality = 0.9
        avg_cost = 0.0
        avg_latency = 0.0
        total_spend = 0.0
        avg_feedback = 0.0

    adjusted_quality = max(0.0, min(1.0, 0.9 + avg_feedback))

    if failure_rate > 0.2:
        action = "demote"
    elif failure_rate > 0.1:
        action = "investigate"
    elif adjusted_quality > 0.92 and failure_rate < 0.05:
        action = "promote"
    else:
        action = "keep"

    metrics = AdaptationMetrics(
        week_start=week_start,
        model_id=model_id,
        total_requests=total_requests,
        failure_count=total_failures,
        failure_rate=failure_rate,
        avg_quality=avg_quality,
        avg_cost=avg_cost,
        avg_latency_ms=avg_latency,
        adjusted_quality_score=adjusted_quality,
        total_spend=total_spend,
        recommended_action=action,
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO adaptation_metrics
            (week_start, model_id, total_requests, failure_count, failure_rate,
             avg_quality, avg_cost, avg_latency_ms, adjusted_quality_score,
             total_spend, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metrics.week_start.isoformat(),
            metrics.model_id,
            metrics.total_requests,
            metrics.failure_count,
            metrics.failure_rate,
            metrics.avg_quality,
            metrics.avg_cost,
            metrics.avg_latency_ms,
            metrics.adjusted_quality_score,
            metrics.total_spend,
            metrics.recommended_action,
        ),
    )
    conn.commit()
    return metrics
