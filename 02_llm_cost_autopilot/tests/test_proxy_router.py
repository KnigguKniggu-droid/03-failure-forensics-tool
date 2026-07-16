"""Tests for the proxy router complexity classifier and model selection."""

from __future__ import annotations

import pytest

from src.models import ComplexityTier, ProxyRequest, ModelConfig, ProviderType
from src.proxy_router import (
    classify_complexity,
    estimate_tokens,
    select_model,
    estimate_cost,
)


def _make_request(messages: list[dict[str, str]], max_tokens: int = 1000) -> ProxyRequest:
    return ProxyRequest(messages=messages, max_tokens=max_tokens)


def test_tier1_simple_extraction():
    req = _make_request([{"role": "user", "content": "Summarize this: Hello world."}])
    result = classify_complexity(req)
    assert result.tier == ComplexityTier.TIER_1_EXTRACTION


def test_tier2_classification():
    req = _make_request([
        {"role": "user", "content": "Classify this email as billing, technical, or general: I need a refund."}
    ])
    result = classify_complexity(req)
    assert result.tier == ComplexityTier.TIER_2_CLASSIFICATION


def test_tier3_code_generation():
    req = _make_request([
        {"role": "user", "content": "Write a Python function def fibonacci(n) that returns the nth Fibonacci number."}
    ])
    result = classify_complexity(req)
    assert result.tier == ComplexityTier.TIER_3_MULTI_STEP_LOGIC


def test_tier3_multi_step_reasoning():
    req = _make_request([
        {"role": "user", "content": "First analyze the dataset, then explain why the results differ, and finally derive a conclusion step by step."}
    ], max_tokens=2000)
    result = classify_complexity(req)
    assert result.tier == ComplexityTier.TIER_3_MULTI_STEP_LOGIC


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 100) == 25
    assert estimate_tokens("a" * 4) == 1


def _make_model(model_id: str, quality: float, cost: float, tiers: list[ComplexityTier]) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider=ProviderType.OPENAI,
        display_name=model_id,
        input_cost_per_1k=cost,
        output_cost_per_1k=cost,
        max_context=128000,
        max_output=4096,
        latency_p50_ms=500,
        quality_score=quality,
        supported_tiers=tiers,
    )


def test_select_model_picks_preferred():
    registry = {
        "gpt-4o-mini": _make_model("gpt-4o-mini", 0.82, 0.0003, [ComplexityTier.TIER_1_EXTRACTION, ComplexityTier.TIER_2_CLASSIFICATION]),
        "gpt-4o": _make_model("gpt-4o", 0.95, 0.01, [ComplexityTier.TIER_1_EXTRACTION, ComplexityTier.TIER_2_CLASSIFICATION, ComplexityTier.TIER_3_MULTI_STEP_LOGIC]),
    }
    rules = {
        "tier_1_preferred": "gpt-4o-mini",
        "tier_2_preferred": "gpt-4o-mini",
        "tier_3_preferred": "gpt-4o",
        "fallback_chain": ["gpt-4o"],
    }
    from src.models import ComplexityClassification
    cls = ComplexityClassification(
        tier=ComplexityTier.TIER_1_EXTRACTION,
        confidence=0.9,
        signals={},
        estimated_input_tokens=100,
        estimated_output_tokens=500,
    )
    model, reason = select_model(cls, registry, rules)
    assert model.model_id == "gpt-4o-mini"
    assert "preferred" in reason


def test_select_model_fallback_to_cost_efficiency():
    registry = {
        "cheap": _make_model("cheap", 0.8, 0.0001, [ComplexityTier.TIER_2_CLASSIFICATION]),
        "expensive": _make_model("expensive", 0.9, 0.01, [ComplexityTier.TIER_2_CLASSIFICATION]),
    }
    rules = {
        "tier_1_preferred": "nonexistent",
        "tier_2_preferred": "nonexistent",
        "tier_3_preferred": "nonexistent",
        "fallback_chain": ["gpt-4o"],
    }
    from src.models import ComplexityClassification
    cls = ComplexityClassification(
        tier=ComplexityTier.TIER_2_CLASSIFICATION,
        confidence=0.8,
        signals={},
        estimated_input_tokens=100,
        estimated_output_tokens=500,
    )
    model, reason = select_model(cls, registry, rules)
    assert model.model_id == "cheap"
    assert "cost-efficiency" in reason


def test_estimate_cost():
    from src.models import ComplexityClassification
    model = _make_model("test", 0.9, 0.005, [ComplexityTier.TIER_1_EXTRACTION])
    cls = ComplexityClassification(
        tier=ComplexityTier.TIER_1_EXTRACTION,
        confidence=0.9,
        signals={},
        estimated_input_tokens=1000,
        estimated_output_tokens=2000,
    )
    cost = estimate_cost(model, cls)
    assert cost == pytest.approx(0.015)
