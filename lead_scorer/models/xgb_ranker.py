"""Calibrated XGBoost classifier used as the production ranker.

XGBoost's raw probabilities are typically over-confident at the extremes. We
wrap it in `CalibratedClassifierCV` (isotonic) so the scores you push to the
CRM are usable as probabilities, not just rank-orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


@dataclass
class XGBRanker:
    """Calibrated XGBoost classifier. Use `predict_proba` for ranking + scoring."""

    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.08
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42
    calibration_method: str = "isotonic"
    calibration_cv: int = 5

    base_: XGBClassifier | None = None
    calibrated_: CalibratedClassifierCV | None = None
    fitted_: bool = False

    def _make_base(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBRanker":
        self.base_ = self._make_base()
        self.calibrated_ = CalibratedClassifierCV(
            estimator=self._make_base(),
            method=self.calibration_method,
            cv=self.calibration_cv,
        )
        # Fit the uncalibrated base too — used by tests that want to compare
        # Brier before vs after calibration.
        self.base_.fit(X, y)
        self.calibrated_.fit(X, y)
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated P(converted=1) as a 1-D array."""
        if not self.fitted_ or self.calibrated_ is None:
            raise RuntimeError("XGBRanker.predict_proba called before fit().")
        return self.calibrated_.predict_proba(X)[:, 1]

    def predict_proba_uncalibrated(self, X: np.ndarray) -> np.ndarray:
        """Raw XGBoost P(converted=1). Useful for calibration diagnostics."""
        if not self.fitted_ or self.base_ is None:
            raise RuntimeError("XGBRanker not fitted.")
        return self.base_.predict_proba(X)[:, 1]
