"""Tests for the calibrated XGBoost ranker."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from lead_scorer.features import FeatureBuilder
from lead_scorer.models import XGBRanker


def test_xgb_fits_and_predicts(leads_df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    X = fb.fit_transform(leads_df)
    y = leads_df["converted"].to_numpy()
    xgb = XGBRanker(n_estimators=80, max_depth=3).fit(X, y)
    probs = xgb.predict_proba(X)
    assert probs.shape == (len(leads_df),)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_calibration_improves_brier(leads_df: pd.DataFrame) -> None:
    """Isotonic calibration should not make Brier worse on a holdout."""
    fb = FeatureBuilder()
    train, test = train_test_split(leads_df, test_size=0.3, random_state=0, stratify=leads_df["converted"])
    X_train = fb.fit_transform(train)
    X_test = fb.transform(test)
    y_train = train["converted"].to_numpy()
    y_test = test["converted"].to_numpy()

    xgb = XGBRanker(n_estimators=120, max_depth=3).fit(X_train, y_train)
    raw_brier = brier_score_loss(y_test, xgb.predict_proba_uncalibrated(X_test))
    cal_brier = brier_score_loss(y_test, xgb.predict_proba(X_test))

    # Calibration should be at least as good (allow tiny tolerance for noise).
    assert cal_brier <= raw_brier + 0.005, (
        f"calibration worsened Brier: raw={raw_brier:.4f}, calibrated={cal_brier:.4f}"
    )
