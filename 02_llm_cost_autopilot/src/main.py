"""FastAPI application for the LLM Cost Autopilot proxy."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models import ProxyRequest, ProxyResponse
from src.proxy_router import (
    classify_complexity,
    load_model_registry,
    load_routing_rules,
    route_and_execute,
)
from src.verifier import get_db_conn, should_verify, verify_response, record_failure

_registry: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _registry["models"] = load_model_registry()
    _registry["rules"] = load_routing_rules()
    _registry["db"] = get_db_conn()
    yield
    _registry["db"].close()


app = FastAPI(
    title="LLM Cost Autopilot",
    description="Intelligent multi-provider routing proxy with cost optimization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions", response_model=ProxyResponse)
async def chat_completions(request: ProxyRequest) -> ProxyResponse:
    """OpenAI-compatible endpoint that routes to the optimal model."""
    try:
        response = await route_and_execute(
            request,
            registry=_registry.get("models"),
            rules=_registry.get("rules"),
        )

        if should_verify(sample_rate=0.1):
            quality = await verify_response(response, request.messages)
            response.verification_score = quality
            model_cfg = _registry["models"].get(response.model)
            if model_cfg and quality < 0.7:
                record_failure(_registry["db"], response, model_cfg, quality, "quality")

        return response
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Routing failure: {exc}")


@app.get("/v1/classify")
async def classify(request: ProxyRequest) -> dict[str, Any]:
    """Return the complexity classification without executing the request."""
    result = classify_complexity(request)
    return {
        "tier": result.tier.value,
        "tier_name": result.tier.name,
        "confidence": result.confidence,
        "signals": result.signals,
        "estimated_input_tokens": result.estimated_input_tokens,
        "estimated_output_tokens": result.estimated_output_tokens,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List all configured models with their cost and capability metadata."""
    models = _registry.get("models", {})
    return {
        model_id: {
            "display_name": cfg.display_name,
            "provider": cfg.provider.value,
            "input_cost_per_1k": cfg.input_cost_per_1k,
            "output_cost_per_1k": cfg.output_cost_per_1k,
            "quality_score": cfg.quality_score,
            "supported_tiers": [t.value for t in cfg.supported_tiers],
            "cost_efficiency": cfg.cost_efficiency,
        }
        for model_id, cfg in models.items()
    }


@app.get("/v1/metrics/{model_id}")
async def get_metrics(model_id: str) -> dict[str, Any]:
    """Get weekly adaptation metrics for a specific model."""
    from src.verifier import compute_weekly_metrics
    metrics = compute_weekly_metrics(_registry["db"], model_id)
    return metrics.model_dump()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
