"""FastAPI application for the Realtime LLM Observability dashboard engine.

Provides WebSocket-based real-time metric streaming and REST endpoints
for dashboard snapshots and historical queries.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path

from src.aggregator import MetricAggregator
from src.collector import MetricCollector
from src.models import MetricType, StreamEvent

app = FastAPI(
    title="Realtime LLM Observability",
    description="Streaming P95 latency and token drift dashboard engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_collector = MetricCollector(window_seconds=60)
_aggregator = MetricAggregator(window_seconds=60)


@app.post("/v1/stream/start")
async def stream_start(request_id: str, model: str) -> dict[str, Any]:
    """Record the start of a streaming inference call."""
    await _collector.record_stream_start(request_id, model)
    return {"status": "started", "request_id": request_id}


@app.post("/v1/stream/chunk")
async def stream_chunk(
    request_id: str,
    model: str,
    chunk_index: int,
    token_count: int = 0,
) -> dict[str, Any]:
    """Record a chunk received during streaming."""
    await _collector.record_stream_chunk(request_id, model, chunk_index, token_count)
    return {"status": "recorded", "chunk_index": chunk_index}


@app.post("/v1/stream/end")
async def stream_end(
    request_id: str,
    model: str,
    total_tokens: int,
    expected_tokens: int = 0,
) -> dict[str, Any]:
    """Record the end of a streaming inference call."""
    await _collector.record_stream_end(request_id, model, total_tokens, expected_tokens)
    return {"status": "completed", "total_tokens": total_tokens}


@app.post("/v1/stream/error")
async def stream_error(request_id: str, model: str, error: str) -> dict[str, Any]:
    """Record a streaming error."""
    await _collector.record_error(request_id, model, error)
    return {"status": "error_recorded"}


@app.get("/v1/dashboard")
async def dashboard_snapshot() -> dict[str, Any]:
    """Get a complete dashboard snapshot of current metrics."""
    snapshot = _aggregator.build_snapshot(_collector)
    return snapshot.model_dump()


@app.get("/v1/metrics/latency")
async def latency_metrics(model: str | None = None) -> dict[str, Any]:
    """Get rolling window latency metrics."""
    metrics = _collector.get_metrics(MetricType.LATENCY, model=model)
    if not metrics:
        return {"p50": 0, "p95": 0, "p99": 0, "sample_count": 0}
    by_model = _aggregator.compute_all_models(metrics, MetricType.LATENCY)
    return {m: r.model_dump() for m, r in by_model.items()}


@app.get("/v1/metrics/token-drift")
async def token_drift_metrics() -> dict[str, Any]:
    """Get token drift metrics and anomalies."""
    drift_metrics = _collector.get_metrics(MetricType.TOKEN_DRIFT)
    anomalies = _aggregator.detect_token_drift_anomalies(drift_metrics)
    rolling = _aggregator.compute_all_models(drift_metrics, MetricType.TOKEN_DRIFT)
    return {
        "rolling": {m: r.model_dump() for m, r in rolling.items()},
        "anomalies": [a.model_dump() for a in anomalies[-20:]],
    }


@app.get("/v1/latency/{request_id}")
async def request_latency_breakdown(request_id: str) -> dict[str, Any]:
    """Get detailed latency breakdown for a specific request."""
    breakdown = _collector.get_latency_breakdown(request_id)
    if breakdown is None:
        return {"error": "request not found"}
    return breakdown.model_dump()


@app.get("/v1/active-streams")
async def active_streams() -> dict[str, Any]:
    """Get the count of currently active streams."""
    return {"active_streams": _collector.active_stream_count}


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()
    try:
        while True:
            snapshot = _aggregator.build_snapshot(_collector)
            await websocket.send_json(snapshot.model_dump())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui() -> str:
    """Serve the real-time dashboard HTML."""
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return dashboard_path.read_text(encoding="utf-8")
    return "<html><body><h1>Realtime LLM Observability</h1><p>Dashboard file not found.</p></body></html>"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
