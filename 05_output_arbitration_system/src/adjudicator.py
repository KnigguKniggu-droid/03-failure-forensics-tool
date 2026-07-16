"""Central Adjudicator node for the arbitration graph.

Aggregates critic outputs into an evidence chain, resolves score
deviations, and produces a final score from 1 to 10.
"""

from __future__ import annotations

import statistics
from typing import Any

from src.models import AdjudicationInput, AdjudicationResult, CriticOutput, CriticType

ADJUDICATOR_SYSTEM_PROMPT = (
    "You are the central adjudicator in an LLM output arbitration system. "
    "You receive scores from three independent critics: Factual Accuracy, "
    "Logical Consistency, and Completeness. Your job is to resolve any "
    "score deviations, weigh the evidence, and produce a final score from "
    "1 to 10. Consider the confidence of each critic and the severity of "
    "any issues raised. If critics disagree significantly, investigate the "
    "evidence to determine which is more credible."
)

WEIGHTS: dict[CriticType, float] = {
    CriticType.FACTUAL_ACCURACY: 0.40,
    CriticType.LOGICAL_CONSISTENCY: 0.35,
    CriticType.COMPLETENESS: 0.25,
}

DEVIATION_THRESHOLD = 2.0
REJECT_THRESHOLD = 4
REVISE_THRESHOLD = 6


def compute_weighted_score(critic_outputs: list[CriticOutput]) -> float:
    """Compute a confidence-weighted average of critic scores."""
    total_weight = 0.0
    weighted_sum = 0.0
    for output in critic_outputs:
        weight = WEIGHTS.get(output.critic_type, 0.33) * output.confidence
        weighted_sum += output.score * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 5.0


def compute_consensus_level(critic_outputs: list[CriticOutput]) -> float:
    """Measure how much the critics agreed (1.0 = perfect agreement)."""
    if len(critic_outputs) < 2:
        return 1.0
    scores = [o.score for o in critic_outputs]
    score_range = max(scores) - min(scores)
    return max(0.0, 1.0 - (score_range / 10.0))


def compute_deviations(
    critic_outputs: list[CriticOutput],
    final_score: float,
) -> dict[str, float]:
    """Compute how much each critic deviates from the final score."""
    return {
        output.critic_type.value: output.score - final_score
        for output in critic_outputs
    }


def build_evidence_chain(critic_outputs: list[CriticOutput]) -> list[str]:
    """Build an ordered evidence chain from all critic outputs."""
    chain: list[str] = []
    for output in critic_outputs:
        chain.append(f"[{output.critic_type.value}] Score: {output.score:.1f}/10 (confidence: {output.confidence:.0%})")
        chain.append(f"  Critique: {output.critique}")
        for ev in output.evidence:
            chain.append(f"  Evidence: {ev}")
    return chain


def determine_verdict(final_score: int) -> str:
    """Determine the verdict based on the final score."""
    if final_score >= REVISE_THRESHOLD + 1:
        return "accept"
    elif final_score >= REJECT_THRESHOLD + 1:
        return "revise"
    else:
        return "reject"


def adjudicate(adjudication_input: AdjudicationInput) -> AdjudicationResult:
    """Run the central adjudicator on critic outputs.

    This is the deterministic adjudication logic. In production, this would
    be augmented with an LLM call to produce natural language reasoning.
    """
    critic_outputs = adjudication_input.critic_outputs
    if not critic_outputs:
        return AdjudicationResult(
            final_score=1,
            verdict="reject",
            reasoning="No critic outputs provided",
            evidence_chain=[],
            critic_scores={},
            score_deviations={},
            consensus_level=0.0,
        )

    weighted = compute_weighted_score(critic_outputs)
    final_score = max(1, min(10, round(weighted)))
    consensus = compute_consensus_level(critic_outputs)
    deviations = compute_deviations(critic_outputs, float(final_score))
    evidence_chain = build_evidence_chain(critic_outputs)
    verdict = determine_verdict(final_score)

    critic_scores = {
        output.critic_type.value: output.score
        for output in critic_outputs
    }

    deviation_notes = []
    for critic_type_str, deviation in deviations.items():
        if abs(deviation) > DEVIATION_THRESHOLD:
            deviation_notes.append(
                f"{critic_type_str} deviates by {deviation:+.1f} from final score"
            )

    reasoning_parts = [
        f"Weighted score: {weighted:.2f} (rounded to {final_score})",
        f"Consensus level: {consensus:.0%}",
        f"Verdict: {verdict}",
    ]
    if deviation_notes:
        reasoning_parts.append("Score deviations: " + "; ".join(deviation_notes))
    reasoning_parts.append(f"Based on {len(critic_outputs)} critic evaluations")

    return AdjudicationResult(
        final_score=final_score,
        verdict=verdict,
        reasoning=". ".join(reasoning_parts),
        evidence_chain=evidence_chain,
        critic_scores=critic_scores,
        score_deviations=deviations,
        consensus_level=consensus,
    )
