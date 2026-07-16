"""Async batch scoring engine for regression detection.

Runs model predictions in concurrent batches, computes binary accuracy
against ground-truth labels, and aggregates per-difficulty metrics.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from src.models import (
    EmailCategory,
    ExpectedDifficulty,
    GroundTruthItem,
    ModelPrediction,
    PromptConfig,
    RegressionReport,
    RegressionSeverity,
    ScoringResult,
)

# Thresholds for regression severity classification
WARNING_THRESHOLD = 0.03  # >3% regression triggers warning
CRITICAL_THRESHOLD = 0.08  # >8% regression triggers critical (blocks merge)
BATCH_SIZE = 5
MAX_CONCURRENT = 10


class LLMClient:
    """Thin async wrapper around the OpenAI Chat Completions API."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openai.com/v1") -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def classify(
        self,
        config: PromptConfig,
        subject: str,
        body: str,
        client: httpx.AsyncClient | None = None,
    ) -> ModelPrediction:
        user_content = config.user_template.format(subject=subject, body=body)
        payload: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        own_client = client is None
        cli = client or httpx.AsyncClient(timeout=30.0)
        start = time.monotonic()
        try:
            resp = await cli.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            latency = (time.monotonic() - start) * 1000
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip().lower()
            tokens = data.get("usage", {}).get("total_tokens", 0)
            predicted = _parse_category(raw, config.categories)
            return ModelPrediction(
                item_id="",
                predicted_category=predicted,
                raw_output=raw,
                latency_ms=latency,
                token_count=tokens,
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return ModelPrediction(
                item_id="",
                raw_output="",
                latency_ms=latency,
                error=str(exc),
            )
        finally:
            if own_client:
                await cli.aclose()


def _parse_category(raw: str, allowed: list[EmailCategory]) -> EmailCategory | None:
    raw_clean = raw.strip().strip(".")
    for cat in allowed:
        if raw_clean == cat.value or raw_clean.startswith(cat.value):
            return cat
    return None


def load_ground_truth(path: Path) -> list[GroundTruthItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GroundTruthItem.model_validate(item) for item in data["test_items"]]


async def run_predictions_batched(
    config: PromptConfig,
    items: list[GroundTruthItem],
    llm_client: LLMClient,
) -> list[ModelPrediction]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _predict(item: GroundTruthItem) -> ModelPrediction:
        async with semaphore:
            pred = await llm_client.classify(config, item.subject, item.body)
            pred.item_id = item.id
            return pred

    tasks = [_predict(item) for item in items]
    return await asyncio.gather(*tasks)


def compute_accuracy(
    items: list[GroundTruthItem], predictions: list[ModelPrediction]
) -> tuple[int, int]:
    correct = 0
    for item, pred in zip(items, predictions, strict=True):
        if pred.predicted_category == item.expected_category:
            correct += 1
    return correct, len(items)


def compute_per_difficulty(
    items: list[GroundTruthItem], predictions: list[ModelPrediction]
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[bool]] = {}
    for item, pred in zip(items, predictions, strict=True):
        diff = item.expected_difficulty.value
        groups.setdefault(diff, []).append(pred.predicted_category == item.expected_category)
    result: dict[str, dict[str, float]] = {}
    for diff, results in groups.items():
        total = len(results)
        correct = sum(results)
        result[diff] = {
            "total": float(total),
            "correct": float(correct),
            "accuracy": correct / total if total > 0 else 0.0,
        }
    return result


def classify_severity(regression_delta: float) -> RegressionSeverity:
    if regression_delta <= -CRITICAL_THRESHOLD:
        return RegressionSeverity.CRITICAL
    if regression_delta <= -WARNING_THRESHOLD:
        return RegressionSeverity.WARNING
    return RegressionSeverity.NONE


def build_report(
    config: PromptConfig,
    items: list[GroundTruthItem],
    predictions: list[ModelPrediction],
    scoring_results: list[ScoringResult],
    baseline_accuracy: float,
    baseline_judge: float,
) -> RegressionReport:
    correct, total = compute_accuracy(items, predictions)
    accuracy = correct / total if total > 0 else 0.0
    regression_delta = accuracy - baseline_accuracy
    severity = classify_severity(regression_delta)
    mean_judge = sum(r.judge_relevance_score for r in scoring_results) / len(scoring_results) if scoring_results else 0.0
    failed = [r for r in scoring_results if not r.is_correct]
    per_diff = compute_per_difficulty(items, predictions)

    return RegressionReport(
        run_id=str(uuid.uuid4()),
        prompt_id=config.prompt_id,
        prompt_version=config.version,
        model=config.model,
        total_items=total,
        correct_count=correct,
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        regression_delta=regression_delta,
        severity=severity,
        mean_judge_relevance=mean_judge,
        per_difficulty=per_diff,
        failed_items=failed,
        statistical_significance={
            "sample_size": total,
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": accuracy,
            "delta": regression_delta,
            "warning_threshold": WARNING_THRESHOLD,
            "critical_threshold": CRITICAL_THRESHOLD,
        },
    )
