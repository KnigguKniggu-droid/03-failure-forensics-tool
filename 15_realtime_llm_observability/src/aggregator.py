"""Rolling window metric aggregator for P95 latency and token drift.

Computes percentile-based metrics over rolling time windows and detects
token count drift anomalies for the real-time dashboard.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from src.models import (
    DashboardSnapshot,
    MetricType,
    RollingWindowMetric,
    StreamingMetric,
    TokenDriftMetric,
)


class MetricAggregator:
    """Aggregates streaming metrics into rolling window percentiles."""

    def __init__(self, window_seconds: int = 60) -> None:
        self.window_seconds = window_seconds
        self._drift_threshold = 0.2
        self._anomaly_buffer: list[TokenDriftMetric] = []

    def compute_rolling_window(
        self,
        metrics: list[StreamingMetric],
        metric_type: MetricType,
        model: str,
    ) -> RollingWindowMetric:
        """Compute rolling window statistics for a specific metric type and model."""
        values = np.array([m.value for m in metrics], dtype=np.float64)

        if len(values) == 0:
            return RollingWindowMetric(
                metric_type=metric_type,
                model=model,
                window_seconds=self.window_seconds,
            )

        return RollingWindowMetric(
            metric_type=metric_type,
            model=model,
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            min_val=float(np.min(values)),
            max_val=float(np.max(values)),
            sample_count=len(values),
            window_seconds=self.window_seconds,
        )

    def compute_all_models(
        self,
        collector_metrics: list[StreamingMetric],
        metric_type: MetricType,
    ) -> dict[str, RollingWindowMetric]:
        """Compute rolling window metrics for all models for a given metric type."""
        by_model: dict[str, list[StreamingMetric]] = defaultdict(list)
        for m in collector_metrics:
            if m.metric_type == metric_type:
                by_model[m.model].append(m)

        return {
            model: self.compute_rolling_window(metrics, metric_type, model)
            for model, metrics in by_model.items()
        }

    def detect_token_drift_anomalies(
        self,
        drift_metrics: list[StreamingMetric],
    ) -> list[TokenDriftMetric]:
        """Detect token count drift anomalies from streaming metrics."""
        anomalies: list[TokenDriftMetric] = []
        for m in drift_metrics:
            if m.metric_type != MetricType.TOKEN_DRIFT:
                continue
            drift_ratio = m.value
            is_anomaly = abs(drift_ratio - 1.0) > self._drift_threshold
            direction = "stable"
            if drift_ratio > 1.0 + self._drift_threshold:
                direction = "over"
            elif drift_ratio < 1.0 - self._drift_threshold:
                direction = "under"

            if is_anomaly:
                td = TokenDriftMetric(
                    request_id=m.request_id,
                    model=m.model,
                    expected_tokens=int(m.metadata.get("expected", 0)),
                    actual_tokens=int(m.metadata.get("actual", 0)),
                    drift_ratio=drift_ratio,
                    drift_direction=direction,
                    is_anomaly=True,
                    threshold=self._drift_threshold,
                    timestamp=m.timestamp,
                )
                anomalies.append(td)
                self._anomaly_buffer.append(td)

        if len(self._anomaly_buffer) > 100:
            self._anomaly_buffer = self._anomaly_buffer[-100:]

        return anomalies

    def compute_model_health(
        self,
        latency_metrics: dict[str, RollingWindowMetric],
        error_metrics: dict[str, RollingWindowMetric],
    ) -> dict[str, dict[str, Any]]:
        """Compute health status for each model."""
        health: dict[str, dict[str, Any]] = {}
        all_models = set(latency_metrics.keys()) | set(error_metrics.keys())

        for model in all_models:
            latency = latency_metrics.get(model)
            errors = error_metrics.get(model)

            p95_latency = latency.p95 if latency else 0.0
            error_rate = errors.mean if errors else 0.0
            sample_count = latency.sample_count if latency else 0

            if error_rate > 0.1 or p95_latency > 10000:
                status = "critical"
            elif error_rate > 0.05 or p95_latency > 5000:
                status = "warning"
            else:
                status = "healthy"

            health[model] = {
                "status": status,
                "p95_latency_ms": p95_latency,
                "error_rate": error_rate,
                "sample_count": sample_count,
            }

        return health

    def build_snapshot(
        self,
        collector: Any,
    ) -> DashboardSnapshot:
        """Build a complete dashboard snapshot from the metric collector."""
        all_metrics = collector.get_metrics(window_seconds=self.window_seconds)

        latency_metrics = self.compute_all_models(
            [m for m in all_metrics if m.metric_type == MetricType.LATENCY and m.metadata.get("type") == "total"],
            MetricType.LATENCY,
        )
        drift_raw = [m for m in all_metrics if m.metric_type == MetricType.TOKEN_DRIFT]
        drift_metrics = self.compute_all_models(drift_raw, MetricType.TOKEN_DRIFT)
        throughput_metrics = self.compute_all_models(
            [m for m in all_metrics if m.metric_type == MetricType.TOKEN_COUNT],
            MetricType.THROUGHPUT,
        )
        error_metrics = self.compute_all_models(
            [m for m in all_metrics if m.metric_type == MetricType.ERROR_RATE],
            MetricType.ERROR_RATE,
        )

        anomalies = self.detect_token_drift_anomalies(drift_raw)
        model_health = self.compute_model_health(latency_metrics, error_metrics)

        total_requests = sum(m.sample_count for m in latency_metrics.values())
        overall_error_rate = (
            sum(m.mean * m.sample_count for m in error_metrics.values()) / max(1, total_requests)
            if error_metrics else 0.0
        )

        return DashboardSnapshot(
            latency_metrics=latency_metrics,
            token_drift_metrics=drift_metrics,
            throughput_metrics=throughput_metrics,
            active_streams=collector.active_stream_count,
            total_requests_window=total_requests,
            error_rate=overall_error_rate,
            anomalies=anomalies[-20:],
            model_health=model_health,
        )
