"""Logistic-regression baseline.

This isn't the production scorer — it's the interpretable counterpart that
sales managers can read coefficients off. If the LR and XGBoost rank-orders
disagree dramatically, that's a signal something is off (drift, leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LRBaseline:
    """Standardised logistic regression with L2 regularisation."""

    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42

    scaler_: StandardScaler = field(default_factory=StandardScaler)
    model_: LogisticRegression | None = None
    fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LRBaseline":
        X_s = self.scaler_.fit_transform(X)
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model_.fit(X_s, y)
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(converted=1) as a 1-D array in [0, 1]."""
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("LRBaseline.predict_proba called before fit().")
        X_s = self.scaler_.transform(X)
        return self.model_.predict_proba(X_s)[:, 1]

    @property
    def coefficients(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("LRBaseline has not been fit.")
        return self.model_.coef_.ravel()
