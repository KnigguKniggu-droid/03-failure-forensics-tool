"""Background Celery pipeline for LLM-as-judge quality monitoring.

Queries downstream output blocks using an LLM-as-judge evaluation window
and feeds results back into the rollout engine.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.models import FeatureFlag, QualityMetric

try:
    from celery import Celery, Task
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False
    Task = object  # type: ignore

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

if HAS_CELERY:
    app = Celery("feature_flag_monitor", broker=REDIS_URL, backend=REDIS_URL)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        beat_schedule={
            "evaluate-canary-quality": {
                "task": "src.celery_tasks.evaluate_canary_quality_task",
                "schedule": 60.0,
            },
        },
    )
else:
    app = None


JUDGE_PROMPT = """You are a quality evaluator for AI feature flag canary monitoring.
Evaluate the output below and score it from 0.0 to 1.0 on quality.

Output to evaluate: {output}
Baseline output: {baseline}

Respond in JSON: {{"score": float, "reason": "brief explanation"}}"""


async def _run_judge(output: str, baseline: str, api_key: str) -> dict[str, Any]:
    """Run the LLM-as-judge evaluation."""
    import httpx

    prompt = JUDGE_PROMPT.format(output=output[:1000], baseline=baseline[:1000])
    payload = {
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a strict quality evaluator."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception:
        return {"score": 0.5, "reason": "Judge evaluation failed"}


def evaluate_quality_metric(
    metric_name: str,
    canary_outputs: list[str],
    baseline_outputs: list[str],
    baseline_score: float = 0.9,
    threshold: float = 0.05,
) -> QualityMetric:
    """Evaluate a single quality metric using LLM-as-judge."""
    import asyncio

    api_key = os.environ.get("OPENAI_API_KEY", "")
    scores: list[float] = []

    for canary_out, baseline_out in zip(canary_outputs, baseline_outputs, strict=False):
        try:
            result = asyncio.run(_run_judge(canary_out, baseline_out, api_key))
            scores.append(max(0.0, min(1.0, float(result.get("score", 0.5)))))
        except Exception:
            scores.append(0.5)

    canary_score = sum(scores) / len(scores) if scores else 0.5
    is_passing = (baseline_score - canary_score) < threshold

    return QualityMetric(
        name=metric_name,
        baseline_score=baseline_score,
        canary_score=canary_score,
        threshold=threshold,
        sample_count=len(scores),
        is_passing=is_passing,
    )


if HAS_CELERY:

    @app.task(bind=True, base=Task)
    def evaluate_canary_quality_task(self, flag_id: str, canary_outputs: list[str], baseline_outputs: list[str]) -> dict[str, Any]:
        """Celery task to evaluate canary quality metrics."""
        metric = evaluate_quality_metric(
            "output_quality",
            canary_outputs,
            baseline_outputs,
        )
        return metric.model_dump()

    @app.task(bind=True, base=Task)
    def check_error_spike_task(self, flag_id: str, error_count: int, threshold: float) -> dict[str, Any]:
        """Celery task to check for error spikes and trigger rollback."""
        if error_count >= threshold:
            return {"flag_id": flag_id, "rollback_triggered": True, "error_count": error_count}
        return {"flag_id": flag_id, "rollback_triggered": False, "error_count": error_count}
