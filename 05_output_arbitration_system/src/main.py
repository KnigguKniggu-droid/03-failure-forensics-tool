"""FastAPI application for the LLM Output Arbitration System."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.graph import run_arbitration
from src.models import ArbitrationRequest, ArbitrationResponse

app = FastAPI(
    title="LLM Output Arbitration System",
    description="Parallel multi-critic judge and adjudicator for LLM outputs",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/arbitrate", response_model=ArbitrationResponse)
async def arbitrate(request: ArbitrationRequest) -> ArbitrationResponse:
    """Run arbitration on a query-response pair."""
    try:
        return await run_arbitration(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Arbitration failed: {exc}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
