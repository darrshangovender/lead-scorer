"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from lead_scorer.evaluation import evaluate, precision_at_top_k_fraction


def test_top_decile_precision_contrived_perfect() -> None:
    """When score order matches label order, top-decile precision is 1.0."""
    n = 100
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(1.0, 0.0, n)  # highest scores match the positives
    assert precision_at_top_k_fraction(y_true, y_score, 0.10) == 1.0


def test_top_decile_precision_contrived_worst() -> None:
    """When score order is anti-correlated with labels, top-decile precision is 0.0."""
    n = 100
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(0.0, 1.0, n)  # highest scores are the negatives
    assert precision_at_top_k_fraction(y_true, y_score, 0.10) == 0.0


def test_top_decile_precision_random_within_bounds() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.binomial(1, 0.2, size=n)
    y_score = rng.random(n)
    p = precision_at_top_k_fraction(y_true, y_score, 0.10)
    assert 0.0 <= p <= 1.0


def test_top_k_uses_rounded_count() -> None:
    """50 items at 10% -> top 5 items."""
    y_true = np.array([1, 1, 1, 1, 1] + [0] * 45)
    y_score = np.arange(50, 0, -1, dtype=float)  # descending, matches positives
    assert precision_at_top_k_fraction(y_true, y_score, 0.10) == 1.0


def test_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError):
        precision_at_top_k_fraction(np.array([0, 1]), np.array([0.1, 0.9]), 0.0)
    with pytest.raises(ValueError):
        precision_at_top_k_fraction(np.array([0, 1]), np.array([0.1, 0.9]), 1.5)


def test_evaluate_returns_all_metrics() -> None:
    rng = np.random.default_rng(1)
    n = 500
    y_true = rng.binomial(1, 0.25, size=n)
    y_score = rng.random(n)
    m = evaluate(y_true, y_score)
    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0
    assert 0.0 <= m.top_decile_precision <= 1.0
    assert 0.0 <= m.top_quintile_precision <= 1.0
    assert m.n_samples == n
    assert m.n_positives == int(y_true.sum())
