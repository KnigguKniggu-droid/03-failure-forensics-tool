"""LLM-as-judge evaluator for relevance scoring.

Uses a stronger model (gpt-4o) to independently judge whether the
classifier's predicted category is a reasonable classification for the
given email, producing a relevance score from 0.0 to 1.0.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from src.models import EmailCategory, ExpectedDifficulty, GroundTruthItem, ModelPrediction, ScoringResult

JUDGE_MODEL = "gpt-4o"
JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating the quality of an email classification.
You will receive the email subject, body, the expected category, and the predicted category.
Score the prediction's relevance from 0.0 to 1.0 where:
- 1.0 = Perfect match, the predicted category is exactly correct.
- 0.5 = Partially correct, the prediction is plausible but not the best fit.
- 0.0 = Completely wrong category.

Respond in JSON only with this schema:
{"relevance_score": float, "explanation": "brief reason"}"""

JUDGE_USER_TEMPLATE = """Email subject: {subject}
Email body: {body}
Expected category: {expected}
Predicted category: {predicted}
Difficulty: {difficulty}"""


class JudgeResult(BaseModel, extra="forbid"):  # type: ignore
    """Parsed judge response."""
    relevance_score: float
    explanation: str


async def judge_single(
    item: GroundTruthItem,
    prediction: ModelPrediction,
    api_key: str | None,
    base_url: str = "https://api.openai.com/v1",
    client: httpx.AsyncClient | None = None,
) -> ScoringResult:
    predicted_str = prediction.predicted_category.value if prediction.predicted_category else "UNKNOWN"
    user_content = JUDGE_USER_TEMPLATE.format(
        subject=item.subject,
        body=item.body,
        expected=item.expected_category.value,
        predicted=predicted_str,
        difficulty=item.expected_difficulty.value,
    )
    payload: dict[str, Any] = {
        "model": JUDGE_MODEL,
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    own = client is None
    cli = client or httpx.AsyncClient(timeout=30.0)
    try:
        resp = await cli.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        score = float(parsed.get("relevance_score", 0.0))
        explanation = parsed.get("explanation", "")
    except Exception:
        score = 0.0
        explanation = "Judge evaluation failed"
    finally:
        if own:
            await cli.aclose()

    is_correct = prediction.predicted_category == item.expected_category
    return ScoringResult(
        item_id=item.id,
        expected_category=item.expected_category,
        predicted_category=prediction.predicted_category,
        is_correct=is_correct,
        judge_relevance_score=max(0.0, min(1.0, score)),
        judge_explanation=explanation,
        difficulty=item.expected_difficulty,
    )


async def judge_batch(
    items: list[GroundTruthItem],
    predictions: list[ModelPrediction],
    api_key: str | None,
    base_url: str = "https://api.openai.com/v1",
) -> list[ScoringResult]:
    semaphore = asyncio.Semaphore(5)

    async def _judge(item: GroundTruthItem, pred: ModelPrediction) -> ScoringResult:
        async with semaphore:
            return await judge_single(item, pred, api_key, base_url)

    tasks = [
        _judge(item, pred)
        for item, pred in zip(items, predictions, strict=True)
    ]
    return await asyncio.gather(*tasks)
