"""Statistical analysis engine using scipy.stats.

Runs two-sample t-test calculations to declare statistical winners
and fires automated kill switches when performance variations collapse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.models import (
    Experiment,
    ExperimentOutcome,
    KillSwitchResult,
    StatisticalResult,
)


def compute_statistics(
    experiment_id: str,
    control_outcomes: list[ExperimentOutcome],
    treatment_outcomes: list[ExperimentOutcome],
    significance_level: float = 0.05,
) -> StatisticalResult:
    """Run a two-sample t-test on control vs treatment outcomes."""
    control_scores = np.array([o.score for o in control_outcomes])
    treatment_scores = np.array([o.score for o in treatment_outcomes])

    control_mean = float(np.mean(control_scores)) if len(control_scores) > 0 else 0.0
    treatment_mean = float(np.mean(treatment_scores)) if len(treatment_scores) > 0 else 0.0
    control_std = float(np.std(control_scores, ddof=1)) if len(control_scores) > 1 else 0.0
    treatment_std = float(np.std(treatment_scores, ddof=1)) if len(treatment_scores) > 1 else 0.0

    if HAS_SCIPY and len(control_scores) > 1 and len(treatment_scores) > 1:
        t_stat, p_value = sp_stats.ttest_ind(control_scores, treatment_scores, equal_var=False)
        t_stat = float(t_stat)
        p_value = float(p_value)

        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        effect_size = float(abs(treatment_mean - control_mean) / pooled_std) if pooled_std > 0 else 0.0

        se = np.sqrt(control_std**2 / len(control_scores) + treatment_std**2 / len(treatment_scores))
        if se > 0:
            ci_low = float(treatment_mean - control_mean - 1.96 * se)
            ci_high = float(treatment_mean - control_mean + 1.96 * se)
        else:
            ci_low = ci_high = 0.0
    else:
        t_stat = 0.0
        p_value = 1.0
        effect_size = 0.0
        ci_low = ci_high = 0.0

    is_significant = p_value < significance_level and abs(effect_size) > 0.1

    if is_significant:
        winner = "treatment" if treatment_mean > control_mean else "control"
    else:
        winner = "inconclusive"

    return StatisticalResult(
        experiment_id=experiment_id,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        control_std=control_std,
        treatment_std=treatment_std,
        control_n=len(control_outcomes),
        treatment_n=len(treatment_outcomes),
        t_statistic=t_stat,
        p_value=p_value,
        is_significant=is_significant,
        confidence_interval=(ci_low, ci_high),
        winner=winner,
        effect_size=effect_size,
    )


def evaluate_kill_switch(
    experiment: Experiment,
    control_outcomes: list[ExperimentOutcome],
    treatment_outcomes: list[ExperimentOutcome],
) -> KillSwitchResult:
    """Evaluate whether the kill switch should be triggered.

    Fires when the performance delta between treatment and control
    collapses below the experiment's kill_switch_threshold, indicating
    the treatment is performing significantly worse.
    """
    if not control_outcomes or not treatment_outcomes:
        return KillSwitchResult(
            experiment_id=experiment.experiment_id,
            triggered=False,
            reason="Insufficient data for kill switch evaluation",
            performance_delta=0.0,
            threshold=experiment.kill_switch_threshold,
        )

    control_mean = sum(o.score for o in control_outcomes) / len(control_outcomes)
    treatment_mean = sum(o.score for o in treatment_outcomes) / len(treatment_outcomes)
    delta = treatment_mean - control_mean

    triggered = delta < -experiment.kill_switch_threshold
    reason = ""
    variant_killed = ""

    if triggered:
        reason = f"Treatment underperforming by {abs(delta):.4f} (threshold: {experiment.kill_switch_threshold})"
        treatment_var = next(
            (v for v in experiment.variants if v.allocation == VariantAllocation.TREATMENT),
            None,
        )
        variant_killed = treatment_var.variant_id if treatment_var else ""

    return KillSwitchResult(
        experiment_id=experiment.experiment_id,
        triggered=triggered,
        reason=reason,
        performance_delta=delta,
        threshold=experiment.kill_switch_threshold,
        variant_killed=variant_killed,
    )


from src.models import VariantAllocation  # noqa: E402
