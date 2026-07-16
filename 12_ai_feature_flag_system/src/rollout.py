"""Progressive rollout engine with canary quality monitoring.

Automates progressive rollout schedules that advance variants dynamically
only when canary benchmarks match baseline thresholds. Executes immediate
rollbacks when error counters spike.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.models import (
    ErrorSpike,
    FeatureFlag,
    FlagStatus,
    QualityMetric,
    RolloutDecision,
)

ROLLBACK_ERROR_MULTIPLIER = 2.0


def evaluate_canary_quality(metrics: list[QualityMetric]) -> bool:
    """Check if all canary metrics are passing their thresholds."""
    return all(m.is_passing for m in metrics)


def compute_error_rate(error_count: int, window_size: int) -> float:
    """Compute the error rate from a window of observations."""
    if window_size == 0:
        return 0.0
    return error_count / window_size


def check_error_spike(
    flag: FeatureFlag,
    error_count: int,
) -> ErrorSpike | None:
    """Check if the error count exceeds the rollback threshold."""
    error_rate = compute_error_rate(error_count, flag.error_count_window)
    if error_rate >= flag.error_threshold:
        return ErrorSpike(
            flag_id=flag.flag_id,
            error_count=error_count,
            error_rate=error_rate,
            threshold=flag.error_threshold,
            window_size=flag.error_count_window,
        )
    return None


def evaluate_rollout(
    flag: FeatureFlag,
    canary_metrics: list[QualityMetric] | None = None,
    error_count: int = 0,
) -> RolloutDecision:
    """Evaluate whether to advance, hold, or rollback the rollout.

    Advancement only occurs when:
    1. Canary metrics are passing (all meet baseline thresholds)
    2. Error rate is below the rollback threshold
    3. Current percentage is below the target

    Rollback occurs when error rate exceeds the threshold.
    """
    metrics = canary_metrics or flag.canary_metrics

    spike = check_error_spike(flag, error_count)
    if spike and flag.auto_rollback:
        return RolloutDecision(
            flag_id=flag.flag_id,
            action="rollback",
            current_percentage=flag.rollout_percentage,
            new_percentage=0.0,
            reason=f"Error spike: {spike.error_rate:.2%} >= {spike.threshold:.2%} threshold",
            metrics_snapshot={"error_count": spike.error_count, "error_rate": spike.error_rate},
        )

    if flag.rollout_percentage >= flag.target_percentage:
        return RolloutDecision(
            flag_id=flag.flag_id,
            action="complete",
            current_percentage=flag.rollout_percentage,
            new_percentage=flag.rollout_percentage,
            reason="Target percentage reached",
        )

    if not evaluate_canary_quality(metrics):
        failing = [m.name for m in metrics if not m.is_passing]
        return RolloutDecision(
            flag_id=flag.flag_id,
            action="hold",
            current_percentage=flag.rollout_percentage,
            new_percentage=flag.rollout_percentage,
            reason=f"Canary metrics failing: {', '.join(failing)}",
            metrics_snapshot={m.name: m.canary_score for m in metrics},
        )

    new_pct = min(flag.target_percentage, flag.rollout_percentage + flag.incremental_step)

    return RolloutDecision(
        flag_id=flag.flag_id,
        action="advance",
        current_percentage=flag.rollout_percentage,
        new_percentage=new_pct,
        reason=f"Canary passing, advancing from {flag.rollout_percentage:.0%} to {new_pct:.0%}",
        metrics_snapshot={m.name: m.canary_score for m in metrics},
    )


def apply_rollout_decision(flag: FeatureFlag, decision: RolloutDecision) -> FeatureFlag:
    """Apply a rollout decision to a feature flag."""
    flag.rollout_percentage = decision.new_percentage
    flag.updated_at = datetime.now(timezone.utc)

    if decision.action == "rollback":
        flag.status = FlagStatus.ROLLED_BACK
    elif decision.action == "complete":
        flag.status = FlagStatus.FULLY_ROLLED_OUT
    elif decision.action == "hold":
        flag.status = FlagStatus.PAUSED
    else:
        flag.status = FlagStatus.ACTIVE

    return flag
