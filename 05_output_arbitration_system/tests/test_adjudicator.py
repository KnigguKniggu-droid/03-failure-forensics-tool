"""Tests for the adjudicator scoring logic."""

from __future__ import annotations

import pytest

from src.adjudicator import (
    adjudicate,
    compute_weighted_score,
    compute_consensus_level,
    compute_deviations,
    build_evidence_chain,
    determine_verdict,
)
from src.models import AdjudicationInput, CriticOutput, CriticType


def _make_critic(critic_type: CriticType, score: float, confidence: float = 0.9) -> CriticOutput:
    return CriticOutput(
        critic_type=critic_type,
        score=score,
        confidence=confidence,
        evidence=["evidence point"],
        critique="test critique",
        model_used="test-model",
        latency_ms=100.0,
    )


def test_weighted_score_all_high():
    outputs = [
        _make_critic(CriticType.FACTUAL_ACCURACY, 9.0),
        _make_critic(CriticType.LOGICAL_CONSISTENCY, 9.0),
        _make_critic(CriticType.COMPLETENESS, 9.0),
    ]
    score = compute_weighted_score(outputs)
    assert score == pytest.approx(9.0)


def test_weighted_score_factually_weighted_heavier():
    outputs = [
        _make_critic(CriticType.FACTUAL_ACCURACY, 10.0, confidence=1.0),
        _make_critic(CriticType.LOGICAL_CONSISTENCY, 5.0, confidence=1.0),
        _make_critic(CriticType.COMPLETENESS, 5.0, confidence=1.0),
    ]
    score = compute_weighted_score(outputs)
    assert score > 6.0


def test_consensus_perfect_agreement():
    outputs = [
        _make_critic(CriticType.FACTUAL_ACCURACY, 8.0),
        _make_critic(CriticType.LOGICAL_CONSISTENCY, 8.0),
        _make_critic(CriticType.COMPLETENESS, 8.0),
    ]
    assert compute_consensus_level(outputs) == pytest.approx(1.0)


def test_consensus_total_disagreement():
    outputs = [
        _make_critic(CriticType.FACTUAL_ACCURACY, 0.0),
        _make_critic(CriticType.LOGICAL_CONSISTENCY, 10.0),
        _make_critic(CriticType.COMPLETENESS, 5.0),
    ]
    consensus = compute_consensus_level(outputs)
    assert consensus == pytest.approx(0.0)


def test_deviations_computed():
    outputs = [
        _make_critic(CriticType.FACTUAL_ACCURACY, 8.0),
        _make_critic(CriticType.LOGICAL_CONSISTENCY, 6.0),
        _make_critic(CriticType.COMPLETENESS, 7.0),
    ]
    deviations = compute_deviations(outputs, 7.0)
    assert deviations["factual_accuracy"] == pytest.approx(1.0)
    assert deviations["logical_consistency"] == pytest.approx(-1.0)
    assert deviations["completeness"] == pytest.approx(0.0)


def test_evidence_chain_built():
    outputs = [_make_critic(CriticType.FACTUAL_ACCURACY, 8.0)]
    chain = build_evidence_chain(outputs)
    assert len(chain) >= 3
    assert "factual_accuracy" in chain[0]


def test_verdict_accept():
    assert determine_verdict(8) == "accept"
    assert determine_verdict(10) == "accept"


def test_verdict_revise():
    assert determine_verdict(6) == "revise"
    assert determine_verdict(5) == "revise"


def test_verdict_reject():
    assert determine_verdict(4) == "reject"
    assert determine_verdict(1) == "reject"


def test_adjudicate_full_pipeline():
    outputs = [
        _make_critic(CriticType.FACTUAL_ACCURACY, 9.0, confidence=0.95),
        _make_critic(CriticType.LOGICAL_CONSISTENCY, 8.0, confidence=0.90),
        _make_critic(CriticType.COMPLETENESS, 7.0, confidence=0.85),
    ]
    result = adjudicate(AdjudicationInput(query="test", response="test", critic_outputs=outputs))
    assert 1 <= result.final_score <= 10
    assert result.verdict in ("accept", "revise", "reject")
    assert len(result.evidence_chain) > 0
    assert len(result.critic_scores) == 3
    assert result.consensus_level > 0.0


def test_adjudicate_empty_critics():
    result = adjudicate(AdjudicationInput(query="test", response="test", critic_outputs=[]))
    assert result.final_score == 1
    assert result.verdict == "reject"
