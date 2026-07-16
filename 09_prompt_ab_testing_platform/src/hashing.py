"""Deterministic traffic distribution via fixed hash user session splitting.

Uses a consistent hashing scheme to assign users to experiment variants,
ensuring the same user always gets the same variant across sessions.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.models import Experiment, ExperimentVariant, TrafficAssignment, VariantAllocation

HASH_MODULUS = 10000


def fixed_hash(user_id: str, experiment_id: str) -> int:
    """Compute a deterministic hash for a user-experiment pair.

    The hash is stable across calls: the same user_id and experiment_id
    always produce the same hash value, preserving variant exposure
    continuity across sessions.
    """
    key = f"{experiment_id}:{user_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % HASH_MODULUS


def assign_variant(
    user_id: str,
    experiment: Experiment,
) -> TrafficAssignment:
    """Assign a user to a variant using deterministic fixed hashing.

    The hash value is mapped to a variant based on the cumulative
    traffic percentages of each variant.
    """
    hash_val = fixed_hash(user_id, experiment.experiment_id)
    hash_pct = hash_val / HASH_MODULUS

    cumulative = 0.0
    assigned_variant: ExperimentVariant | None = None
    for variant in experiment.variants:
        cumulative += variant.traffic_percentage
        if hash_pct <= cumulative:
            assigned_variant = variant
            break

    if assigned_variant is None:
        assigned_variant = experiment.variants[-1]

    return TrafficAssignment(
        user_id=user_id,
        experiment_id=experiment.experiment_id,
        variant_id=assigned_variant.variant_id,
        allocation=assigned_variant.allocation,
        hash_value=hash_val,
    )


def verify_assignment_stability(
    user_id: str,
    experiment: Experiment,
    iterations: int = 100,
) -> bool:
    """Verify that the same user always gets the same variant."""
    first = assign_variant(user_id, experiment).variant_id
    for _ in range(iterations - 1):
        if assign_variant(user_id, experiment).variant_id != first:
            return False
    return True


def compute_traffic_distribution(
    experiment: Experiment,
    sample_size: int = 10000,
) -> dict[str, int]:
    """Simulate traffic distribution to verify percentage allocation."""
    counts: dict[str, int] = {v.variant_id: 0 for v in experiment.variants}
    for i in range(sample_size):
        assignment = assign_variant(f"user_{i}", experiment)
        counts[assignment.variant_id] += 1
    return counts
