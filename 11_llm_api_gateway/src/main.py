"""FastAPI application for the LLM API Gateway."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.gateway import LLMGateway
from src.models import GatewayConfig, GatewayRequest, VendorConfig
from src.observability import GatewayMetrics, GatewayTracer

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "gateway.yaml"

def load_config() -> GatewayConfig:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(raw)

_config = load_config()
_gateway = LLMGateway(_config)
_metrics = GatewayMetrics()
_tracer = GatewayTracer()

app = FastAPI(
    title="LLM API Gateway",
    description="Distributed rate limiting, fallback routing, and observability",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(request: GatewayRequest) -> dict[str, Any]:
    """Route a chat completion request through the gateway."""
    with _tracer.start_span("gateway_request"):
        response = await _gateway.route_request(request)
        _metrics.record_request(response)
        return response.model_dump()


@app.get("/v1/gateway/status")
async def gateway_status() -> dict[str, Any]:
    """Get gateway status including circuit breaker states and latency metrics."""
    cb_states = {vid: cb.get_state().model_dump() for vid, cb in _gateway.circuit_breakers.items()}
    latency = _gateway.get_latency_metrics()
    return {
        "vendors": list(_gateway.vendors.keys()),
        "circuit_breakers": cb_states,
        "latency_metrics": latency,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Prometheus metrics endpoint."""
    return _metrics.export()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
