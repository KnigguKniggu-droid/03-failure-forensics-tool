"""Realtime LLM Observability.

Streaming P95 latency and token drift dashboard engine. Collects
real-time metrics from LLM inference streams, aggregates them into
rolling window percentiles, and detects token count drift anomalies.
"""

__version__ = "0.1.0"
