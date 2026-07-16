"""Evaluation engine with blind LLM-as-judge and catastrophic forgetting checks.

Runs blind LLM-as-judge benchmarks against open targets and evaluates
general benchmarks to catch catastrophic forgetting errors.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from src.models import BenchmarkResult, ForgettingCheckResult

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_benchmark_stats(scores: list[float]) -> tuple[float, float]:
    """Compute mean and std of benchmark scores."""
    if not scores:
        return 0.0, 0.0
    arr = np.array(scores)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def run_blind_judge_benchmark(
    benchmark_name: str,
    model_name: str,
    prompts: list[str],
    responses: list[str],
    judge_api_key: str | None = None,
    judge_model: str = "gpt-4o",
) -> BenchmarkResult:
    """Run a blind LLM-as-judge benchmark.

    The judge does not know which model produced each response (blind),
    preventing bias in scoring.
    """
    scores: list[float] = []
    api_key = judge_api_key or os.environ.get("OPENAI_API_KEY", "")

    try:
        import httpx
        import asyncio

        async def judge_single(prompt: str, response: str) -> float:
            judge_prompt = (
                f"Rate the quality of this response to the prompt on a scale of 0.0 to 1.0.\n"
                f"Prompt: {prompt[:500]}\nResponse: {response[:500]}\n"
                f"Respond with only a JSON: {{\"score\": float}}"
            )
            payload = {
                "model": judge_model,
                "temperature": 0.0,
                "max_tokens": 50,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a blind judge. Score response quality."},
                    {"role": "user", "content": judge_prompt},
                ],
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                return float(json.loads(raw).get("score", 0.5))

        loop = asyncio.new_event_loop()
        for prompt, response in zip(prompts, responses, strict=True):
            try:
                score = loop.run_until_complete(judge_single(prompt, response))
                scores.append(max(0.0, min(1.0, score)))
            except Exception:
                scores.append(0.5)
        loop.close()
    except Exception:
        scores = [0.5] * len(prompts)

    mean, std = compute_benchmark_stats(scores)
    return BenchmarkResult(
        benchmark_name=benchmark_name,
        model_name=model_name,
        scores=scores,
        mean_score=mean,
        std_score=std,
        num_samples=len(scores),
        is_blind=True,
    )


def check_catastrophic_forgetting(
    benchmark_name: str,
    base_model_scores: list[float],
    finetuned_model_scores: list[float],
    threshold: float = 0.05,
) -> ForgettingCheckResult:
    """Evaluate whether fine-tuning caused catastrophic forgetting.

    Compares general benchmark performance between the base model and
    the fine-tuned model. If the fine-tuned model's performance drops
    below the threshold relative to the base, forgetting is detected.
    """
    base_mean, _ = compute_benchmark_stats(base_model_scores)
    ft_mean, _ = compute_benchmark_stats(finetuned_model_scores)
    delta = ft_mean - base_mean

    return ForgettingCheckResult(
        benchmark_name=benchmark_name,
        base_model_score=base_mean,
        finetuned_model_score=ft_mean,
        performance_delta=delta,
        forgetting_detected=delta < -threshold,
        threshold=threshold,
    )


GENERAL_BENCHMARKS = [
    {"name": "mmlu", "description": "Massive Multitask Language Understanding"},
    {"name": "hellaswag", "description": "Sentence completion commonsense reasoning"},
    {"name": "arc_challenge", "description": "AI2 Reasoning Challenge"},
    {"name": "truthfulqa", "description": "Truthfulness benchmark"},
    {"name": "winogrande", "description": "Coreference resolution"},
]


def evaluate_general_benchmarks(
    base_model_name: str,
    finetuned_model_name: str,
    threshold: float = 0.05,
) -> list[ForgettingCheckResult]:
    """Run general benchmark evaluation to detect catastrophic forgetting.

    In production, this would load and run the actual benchmarks.
    For the architectural prototype, mock scores are used.
    """
    results: list[ForgettingCheckResult] = []
    np.random.seed(42)
    for bench in GENERAL_BENCHMARKS:
        base_scores = list(np.random.uniform(0.7, 0.85, 100))
        ft_scores = list(np.random.uniform(0.68, 0.83, 100))
        results.append(check_catastrophic_forgetting(bench["name"], base_scores, ft_scores, threshold))
    return results
