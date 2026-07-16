"""Observability layer with OpenTelemetry and Prometheus metrics.

Provides distributed tracing via OpenTelemetry spans and Prometheus
metrics for request count, latency histograms, and circuit breaker state.
"""

from __future__ import annotations

import time
from typing import Any

from src.models import CircuitState, GatewayResponse

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class GatewayMetrics:
    """Prometheus metrics collector for the gateway."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry() if HAS_PROMETHEUS else None
        if HAS_PROMETHEUS:
            self.requests_total = Counter(
                "gateway_requests_total",
                "Total requests routed through gateway",
                ["vendor_id", "model", "status"],
                registry=self.registry,
            )
            self.request_latency = Histogram(
                "gateway_request_latency_seconds",
                "Request latency in seconds",
                ["vendor_id"],
                buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
                registry=self.registry,
            )
            self.circuit_state = Gauge(
                "gateway_circuit_state",
                "Circuit breaker state (0=closed, 1=half_open, 2=open)",
                ["vendor_id"],
                registry=self.registry,
            )
            self.rate_limited_total = Counter(
                "gateway_rate_limited_total",
                "Total rate-limited requests",
                ["vendor_id"],
                registry=self.registry,
            )
            self.fallback_total = Counter(
                "gateway_fallback_total",
                "Total fallback routing events",
                ["from_vendor", "to_vendor"],
                registry=self.registry,
            )

    def record_request(self, response: GatewayResponse) -> None:
        """Record metrics for a completed gateway request."""
        if not HAS_PROMETHEUS:
            return
        status = "success" if not response.error else "error"
        self.requests_total.labels(
            vendor_id=response.vendor_id,
            model=response.model,
            status=status,
        ).inc()
        self.request_latency.labels(vendor_id=response.vendor_id).observe(response.latency_ms / 1000.0)
        if response.rate_limited:
            self.rate_limited_total.labels(vendor_id=response.vendor_id).inc()
        if response.fallback_used:
            self.fallback_total.labels(
                from_vendor="primary",
                to_vendor=response.vendor_id,
            ).inc()

    def update_circuit_state(self, vendor_id: str, state: CircuitState) -> None:
        """Update the circuit breaker gauge."""
        if not HAS_PROMETHEUS:
            return
        state_value = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}.get(state, 0)
        self.circuit_state.labels(vendor_id=vendor_id).set(state_value)

    def export(self) -> str:
        """Export Prometheus metrics in text format."""
        if not HAS_PROMETHEUS:
            return ""
        return generate_latest(self.registry).decode("utf-8")


class GatewayTracer:
    """OpenTelemetry tracer for distributed request tracing."""

    def __init__(self, service_name: str = "llm-gateway") -> None:
        self.service_name = service_name
        self._tracer: Any = None
        if HAS_OTEL:
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self.service_name)

    def start_span(self, name: str) -> Any:
        """Start a tracing span."""
        if self._tracer:
            return self._tracer.start_as_current_span(name)
        from contextlib import nullcontext
        return nullcontext()
