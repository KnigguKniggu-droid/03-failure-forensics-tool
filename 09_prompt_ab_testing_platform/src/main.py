"""FastAPI application for the Prompt A/B Testing Platform."""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.hashing import assign_variant, compute_traffic_distribution
from src.models import (
    Experiment,
    ExperimentOutcome,
    ExperimentStatus,
    ExperimentVariant,
    PromptVersion,
    VariantAllocation,
)
from src.statistics import compute_statistics, evaluate_kill_switch

app = FastAPI(
    title="Prompt A/B Testing Platform",
    description="Prompt versioning, deterministic traffic splitting, and statistical analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_experiments: dict[str, Experiment] = {}
_outcomes: dict[str, list[ExperimentOutcome]] = {}


@app.post("/v1/prompts")
async def create_prompt_version(prompt: PromptVersion) -> dict[str, Any]:
    """Register a new prompt version in the registry."""
    return {"prompt_id": prompt.prompt_id, "version": prompt.version, "status": "registered"}


@app.post("/v1/experiments")
async def create_experiment(
    name: str,
    control_prompt: PromptVersion,
    treatment_prompt: PromptVersion,
    traffic_split: float = 0.5,
) -> dict[str, Any]:
    """Create a new A/B test experiment with control and treatment variants."""
    exp_id = str(uuid.uuid4())
    experiment = Experiment(
        experiment_id=exp_id,
        name=name,
        variants=[
            ExperimentVariant(
                variant_id=f"{exp_id}_control",
                allocation=VariantAllocation.CONTROL,
                prompt_version=control_prompt,
                traffic_percentage=1.0 - traffic_split,
            ),
            ExperimentVariant(
                variant_id=f"{exp_id}_treatment",
                allocation=VariantAllocation.TREATMENT,
                prompt_version=treatment_prompt,
                traffic_percentage=traffic_split,
            ),
        ],
        status=ExperimentStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )
    _experiments[exp_id] = experiment
    _outcomes[exp_id] = []
    return experiment.model_dump()


@app.get("/v1/experiments/{experiment_id}/assign/{user_id}")
async def assign_user(experiment_id: str, user_id: str) -> dict[str, Any]:
    """Assign a user to a variant using deterministic hashing."""
    experiment = _experiments.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    assignment = assign_variant(user_id, experiment)
    return assignment.model_dump()


@app.post("/v1/experiments/{experiment_id}/outcome")
async def record_outcome(
    experiment_id: str,
    variant_id: str,
    user_id: str,
    success: bool,
    score: float,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    """Record an experiment outcome observation."""
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    outcome = ExperimentOutcome(
        experiment_id=experiment_id,
        variant_id=variant_id,
        user_id=user_id,
        success=success,
        score=score,
        latency_ms=latency_ms,
    )
    _outcomes.setdefault(experiment_id, []).append(outcome)
    return {"recorded": True, "total_outcomes": len(_outcomes[experiment_id])}


@app.get("/v1/experiments/{experiment_id}/analyze")
async def analyze_experiment(experiment_id: str) -> dict[str, Any]:
    """Run statistical analysis on experiment outcomes."""
    experiment = _experiments.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    outcomes = _outcomes.get(experiment_id, [])
    control_id = next(v.variant_id for v in experiment.variants if v.allocation == VariantAllocation.CONTROL)
    treatment_id = next(v.variant_id for v in experiment.variants if v.allocation == VariantAllocation.TREATMENT)

    control_outcomes = [o for o in outcomes if o.variant_id == control_id]
    treatment_outcomes = [o for o in outcomes if o.variant_id == treatment_id]

    stats = compute_statistics(experiment_id, control_outcomes, treatment_outcomes, experiment.statistical_significance)
    kill = evaluate_kill_switch(experiment, control_outcomes, treatment_outcomes)

    if kill.triggered and experiment.status == ExperimentStatus.RUNNING:
        experiment.status = ExperimentStatus.KILLED
        experiment.ended_at = datetime.now(timezone.utc)

    return {
        "statistics": stats.model_dump(),
        "kill_switch": kill.model_dump(),
        "experiment_status": experiment.status.value,
    }


@app.get("/v1/experiments/{experiment_id}/distribution")
async def traffic_distribution(experiment_id: str) -> dict[str, Any]:
    """Verify traffic distribution for an experiment."""
    experiment = _experiments.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    dist = compute_traffic_distribution(experiment)
    return {"distribution": dist, "total_sampled": sum(dist.values())}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


from datetime import datetime, timezone  # noqa: E402
