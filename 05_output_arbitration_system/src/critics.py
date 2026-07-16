"""Parallel critic nodes for the arbitration graph.

Three critics run simultaneously:
- Factual Accuracy (GPT-4o)
- Logical Consistency (Claude)
- Completeness (Local Llama via Ollama)

Each critic uses instructor-typed definitions for structured output.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from src.models import CriticInput, CriticOutput, CriticType

CRITIC_PROMPTS: dict[CriticType, str] = {
    CriticType.FACTUAL_ACCURACY: (
        "You are a factual accuracy critic. Evaluate whether the response is "
        "factually correct given the query and context. Check for hallucinations, "
        "incorrect claims, and unsupported statements. Score from 0 to 10 where "
        "10 means perfectly factual. Provide specific evidence for each issue found."
    ),
    CriticType.LOGICAL_CONSISTENCY: (
        "You are a logical consistency critic. Evaluate whether the response is "
        "internally consistent, follows logical reasoning, and does not contradict "
        "itself. Check for logical fallacies, circular reasoning, and gaps in "
        "argumentation. Score from 0 to 10 where 10 means perfectly logical."
    ),
    CriticType.COMPLETENESS: (
        "You are a completeness critic. Evaluate whether the response fully "
        "addresses all aspects of the query. Check for missing information, "
        "unanswered sub-questions, and incomplete explanations. Score from 0 to "
        "10 where 10 means fully complete."
    ),
}

CRITIC_MODELS: dict[CriticType, str] = {
    CriticType.FACTUAL_ACCURACY: "gpt-4o",
    CriticType.LOGICAL_CONSISTENCY: "claude-3-5-sonnet",
    CriticType.COMPLETENESS: "llama3.1-8b-local",
}

CRITIC_PROVIDERS: dict[CriticType, str] = {
    CriticType.FACTUAL_ACCURACY: "openai",
    CriticType.LOGICAL_CONSISTENCY: "anthropic",
    CriticType.COMPLETENESS: "ollama",
}


def _build_critic_prompt(critic_input: CriticInput) -> str:
    return (
        f"Query: {critic_input.query}\n"
        f"Response: {critic_input.response}\n"
        f"Context: {critic_input.context or 'No additional context provided.'}\n\n"
        "Evaluate the response and respond in JSON with this schema:\n"
        '{"score": float, "confidence": float, "evidence": ["point1", "point2"], '
        '"critique": "detailed text"}'
    )


async def _call_openai(model: str, system: str, user: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])


async def _call_anthropic(model: str, system: str, user: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "system": system,
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": user + "\n\nRespond in JSON only."}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        return json.loads(text)


async def _call_ollama(model: str, system: str, user: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user + "\n\nRespond in JSON only."},
        ],
        "temperature": 0.0,
        "stream": False,
        "format": "json",
    }
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post("http://localhost:11434/api/chat", json=payload, headers=headers)
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])


async def run_critic(critic_input: CriticInput) -> CriticOutput:
    """Run a single critic node against the configured provider."""
    critic_type = critic_input.critic_type
    system_prompt = CRITIC_PROMPTS[critic_type]
    user_prompt = _build_critic_prompt(critic_input)
    model = CRITIC_MODELS[critic_type]
    provider = CRITIC_PROVIDERS[critic_type]

    start = time.monotonic()
    try:
        if provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            raw = await _call_openai(model, system_prompt, user_prompt, key)
        elif provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            raw = await _call_anthropic(model, system_prompt, user_prompt, key)
        else:
            raw = await _call_ollama(model, system_prompt, user_prompt)

        latency = (time.monotonic() - start) * 1000
        return CriticOutput(
            critic_type=critic_type,
            score=float(raw.get("score", 5.0)),
            confidence=float(raw.get("confidence", 0.5)),
            evidence=raw.get("evidence", []),
            critique=raw.get("critique", ""),
            model_used=model,
            latency_ms=latency,
        )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return CriticOutput(
            critic_type=critic_type,
            score=5.0,
            confidence=0.0,
            evidence=[],
            critique=f"Critic failed: {exc}",
            model_used=model,
            latency_ms=latency,
        )


async def run_all_critics(query: str, response: str, context: str = "") -> list[CriticOutput]:
    """Fan out to all three critics simultaneously."""
    tasks = []
    for critic_type in CriticType:
        critic_input = CriticInput(
            query=query,
            response=response,
            context=context,
            critic_type=critic_type,
        )
        tasks.append(run_critic(critic_input))
    return await asyncio.gather(*tasks)
