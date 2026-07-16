"""Unit tests for the regression detection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    EmailCategory,
    ExpectedDifficulty,
    GroundTruthItem,
    ModelPrediction,
    PromptConfig,
    RegressionReport,
    RegressionSeverity,
    ScoringResult,
)
from src.regressor import (
    classify_severity,
    compute_accuracy,
    compute_per_difficulty,
    build_report,
    load_ground_truth,
)

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"


SAMPLE_CONFIG = PromptConfig(
    prompt_id="classifier_v1",
    version="1.0.0",
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=50,
    system_prompt="Classify the email.",
    user_template="Subject: {subject}\nBody: {body}",
    categories=[
        EmailCategory.BILLING,
        EmailCategory.TECHNICAL,
        EmailCategory.ACCOUNT,
        EmailCategory.GENERAL,
    ],
)


def _make_item(
    id: str,
    category: EmailCategory,
    difficulty: ExpectedDifficulty = ExpectedDifficulty.EASY,
) -> GroundTruthItem:
    return GroundTruthItem(
        id=id,
        subject="Test subject",
        body="Test body",
        expected_category=category,
        expected_difficulty=difficulty,
    )


def test_prompt_config_validates_version_format():
    with pytest.raises(Exception):
        PromptConfig(
            prompt_id="test",
            version="bad-version",
            model="gpt-4o",
            system_prompt="test",
            user_template="test",
            categories=[EmailCategory.BILLING],
        )


def test_prompt_config_requires_all_categories():
    with pytest.raises(Exception):
        PromptConfig(
            prompt_id="test",
            version="1.0.0",
            model="gpt-4o",
            system_prompt="test",
            user_template="test",
            categories=[EmailCategory.BILLING],
        )


def test_load_ground_truth_returns_10_items():
    items = load_ground_truth(GROUND_TRUTH_PATH)
    assert len(items) == 10
    assert all(isinstance(i, GroundTruthItem) for i in items)


def test_ground_truth_covers_all_categories():
    items = load_ground_truth(GROUND_TRUTH_PATH)
    cats = {i.expected_category for i in items}
    assert cats == {EmailCategory.BILLING, EmailCategory.TECHNICAL, EmailCategory.ACCOUNT, EmailCategory.GENERAL}


def test_compute_accuracy_all_correct():
    items = [_make_item("a", EmailCategory.BILLING), _make_item("b", EmailCategory.TECHNICAL)]
    preds = [
        ModelPrediction(item_id="a", predicted_category=EmailCategory.BILLING),
        ModelPrediction(item_id="b", predicted_category=EmailCategory.TECHNICAL),
    ]
    correct, total = compute_accuracy(items, preds)
    assert correct == 2
    assert total == 2


def test_compute_accuracy_partial():
    items = [_make_item("a", EmailCategory.BILLING), _make_item("b", EmailCategory.TECHNICAL)]
    preds = [
        ModelPrediction(item_id="a", predicted_category=EmailCategory.BILLING),
        ModelPrediction(item_id="b", predicted_category=EmailCategory.GENERAL),
    ]
    correct, total = compute_accuracy(items, preds)
    assert correct == 1
    assert total == 2


def test_compute_per_difficulty():
    items = [
        _make_item("a", EmailCategory.BILLING, ExpectedDifficulty.EASY),
        _make_item("b", EmailCategory.TECHNICAL, ExpectedDifficulty.EASY),
        _make_item("c", EmailCategory.ACCOUNT, ExpectedDifficulty.HARD),
    ]
    preds = [
        ModelPrediction(item_id="a", predicted_category=EmailCategory.BILLING),
        ModelPrediction(item_id="b", predicted_category=EmailCategory.GENERAL),
        ModelPrediction(item_id="c", predicted_category=EmailCategory.ACCOUNT),
    ]
    result = compute_per_difficulty(items, preds)
    assert result["easy"]["accuracy"] == 0.5
    assert result["hard"]["accuracy"] == 1.0


def test_classify_severity_none():
    assert classify_severity(0.0) == RegressionSeverity.NONE
    assert classify_severity(0.05) == RegressionSeverity.NONE


def test_classify_severity_warning():
    assert classify_severity(-0.04) == RegressionSeverity.WARNING
    assert classify_severity(-0.03) == RegressionSeverity.WARNING


def test_classify_severity_critical():
    assert classify_severity(-0.08) == RegressionSeverity.CRITICAL
    assert classify_severity(-0.15) == RegressionSeverity.CRITICAL


def test_build_report_computes_delta():
    items = [_make_item("a", EmailCategory.BILLING)]
    preds = [ModelPrediction(item_id="a", predicted_category=EmailCategory.BILLING)]
    scoring = [
        ScoringResult(
            item_id="a",
            expected_category=EmailCategory.BILLING,
            predicted_category=EmailCategory.BILLING,
            is_correct=True,
            judge_relevance_score=1.0,
            difficulty=ExpectedDifficulty.EASY,
        )
    ]
    report = build_report(SAMPLE_CONFIG, items, preds, scoring, baseline_accuracy=0.9, baseline_judge=0.9)
    assert report.accuracy == 1.0
    assert report.regression_delta == pytest.approx(0.1)
    assert report.severity == RegressionSeverity.NONE
    assert not report.blocks_merge


def test_build_report_detects_critical_regression():
    items = [_make_item("a", EmailCategory.BILLING), _make_item("b", EmailCategory.TECHNICAL)]
    preds = [
        ModelPrediction(item_id="a", predicted_category=EmailCategory.GENERAL),
        ModelPrediction(item_id="b", predicted_category=EmailCategory.GENERAL),
    ]
    scoring = [
        ScoringResult(
            item_id="a",
            expected_category=EmailCategory.BILLING,
            predicted_category=EmailCategory.GENERAL,
            is_correct=False,
            judge_relevance_score=0.0,
            difficulty=ExpectedDifficulty.EASY,
        ),
        ScoringResult(
            item_id="b",
            expected_category=EmailCategory.TECHNICAL,
            predicted_category=EmailCategory.GENERAL,
            is_correct=False,
            judge_relevance_score=0.1,
            difficulty=ExpectedDifficulty.EASY,
        ),
    ]
    report = build_report(SAMPLE_CONFIG, items, preds, scoring, baseline_accuracy=1.0, baseline_judge=1.0)
    assert report.accuracy == 0.0
    assert report.regression_delta == pytest.approx(-1.0)
    assert report.severity == RegressionSeverity.CRITICAL
    assert report.blocks_merge
