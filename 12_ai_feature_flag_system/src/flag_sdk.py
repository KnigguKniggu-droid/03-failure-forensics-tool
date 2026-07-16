"""Client SDK for local flag evaluation using consistent hash schemas.

Evaluates flag states locally without making a network call for every
check. Uses consistent hashing on user_id to determine rollout inclusion.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.models import FeatureFlag, FlagEvaluation, FlagStatus, RolloutStrategy

HASH_MODULUS = 10000


def consistent_hash(flag_id: str, user_id: str) -> int:
    """Compute a deterministic hash for flag-user pair."""
    key = f"{flag_id}:{user_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % HASH_MODULUS


class FeatureFlagSDK:
    """Client-side SDK for local feature flag evaluation."""

    def __init__(self) -> None:
        self._flags: dict[str, FeatureFlag] = {}

    def register_flag(self, flag: FeatureFlag) -> None:
        """Register or update a flag configuration."""
        self._flags[flag.flag_id] = flag

    def evaluate(self, flag_id: str, user_id: str) -> FlagEvaluation:
        """Evaluate a flag for a given user using local consistent hashing.

        The evaluation is deterministic: the same user always gets the
        same result for the same flag until the rollout percentage changes.
        """
        flag = self._flags.get(flag_id)
        if flag is None:
            return FlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=False,
                rollout_percentage=0.0,
                evaluation_reason="flag not found",
            )

        if not flag.enabled or flag.status == FlagStatus.PAUSED:
            return FlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=False,
                rollout_percentage=flag.rollout_percentage,
                evaluation_reason="flag disabled or paused",
            )

        if flag.status == FlagStatus.ROLLED_BACK:
            return FlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=False,
                rollout_percentage=0.0,
                evaluation_reason="flag rolled back",
            )

        if flag.rollout_strategy == RolloutStrategy.INSTANT:
            return FlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=True,
                rollout_percentage=1.0,
                evaluation_reason="instant rollout",
            )

        hash_val = consistent_hash(flag_id, user_id)
        hash_pct = hash_val / HASH_MODULUS
        enabled = hash_pct < flag.rollout_percentage

        return FlagEvaluation(
            flag_id=flag_id,
            user_id=user_id,
            enabled=enabled,
            rollout_percentage=flag.rollout_percentage,
            evaluation_reason=f"hash {hash_pct:.4f} {'<' if enabled else '>='} rollout {flag.rollout_percentage:.4f}",
            hash_value=hash_val,
        )

    def get_all_flags(self) -> list[FeatureFlag]:
        return list(self._flags.values())

    def update_flag(self, flag_id: str, **updates: Any) -> FeatureFlag | None:
        """Update a flag's properties."""
        flag = self._flags.get(flag_id)
        if flag is None:
            return None
        for key, value in updates.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        from datetime import datetime, timezone
        flag.updated_at = datetime.now(timezone.utc)
        return flag
