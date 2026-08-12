"""Tests for the logistic-regression baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_scorer.features import FeatureBuilder
from lead_scorer.models import LRBaseline


def test_lr_fits_and_predicts_shape(leads_df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    X = fb.fit_transform(leads_df)
    y = leads_df["converted"].to_numpy()
    lr = LRBaseline().fit(X, y)
    probs = lr.predict_proba(X)
    assert probs.shape == (len(leads_df),)
    assert probs.ndim == 1


def test_lr_probs_in_unit_interval(leads_df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    X = fb.fit_transform(leads_df)
    y = leads_df["converted"].to_numpy()
    lr = LRBaseline().fit(X, y)
    probs = lr.predict_proba(X)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_lr_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        LRBaseline().predict_proba(np.zeros((1, 5)))


def test_lr_coefficients_shape(leads_df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    X = fb.fit_transform(leads_df)
    y = leads_df["converted"].to_numpy()
    lr = LRBaseline().fit(X, y)
    assert lr.coefficients.shape == (fb.n_features,)
