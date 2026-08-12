"""Feature engineering for raw lead rows.

The `FeatureBuilder` takes the raw schema produced by `sample_generator` and
returns a numeric feature matrix ready for sklearn / XGBoost. It is
fit-then-transform so the category set is locked from training and replayed at
inference — new categories at score time fall through as all-zeros, which
matches how the production pipeline handles unseen values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Columns the builder requires on input. Anything else is ignored.
RAW_REQUIRED: list[str] = [
    "industry",
    "region",
    "company_size",
    "days_since_last_touch",
    "email_engagement_score",
    "page_views_30d",
    "demo_requested",
    "seniority_score",
]

# Categorical columns we one-hot encode.
CATEGORICAL: list[str] = ["industry", "region", "company_size"]

# Numeric columns we pass through.
NUMERIC: list[str] = [
    "days_since_last_touch",
    "email_engagement_score",
    "page_views_30d",
    "demo_requested",
    "seniority_score",
]


@dataclass
class FeatureBuilder:
    """Fits category vocabularies on training data and replays them at scoring.

    Usage:
        fb = FeatureBuilder()
        X_train = fb.fit_transform(train_df)
        X_test = fb.transform(test_df)
    """

    categories_: dict[str, list[str]] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in RAW_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # Required fields must be non-null.
        for c in RAW_REQUIRED:
            if df[c].isna().any():
                raise ValueError(f"Column {c!r} contains nulls; impute upstream.")

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        """Learn category vocabularies from the training frame."""
        self._validate(df)
        self.categories_ = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL}

        names: list[str] = list(NUMERIC)
        for c in CATEGORICAL:
            for v in self.categories_[c]:
                names.append(f"{c}__{v}")
        self.feature_names_ = names
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the learned encoding. Returns a dense float matrix."""
        if not self.fitted_:
            raise RuntimeError("FeatureBuilder.transform called before fit().")
        self._validate(df)

        # Numeric block.
        num = df[NUMERIC].astype(float).to_numpy()

        # One-hot block — locked vocab, unseen values fall through as zero rows.
        ohe_blocks: list[np.ndarray] = []
        for c in CATEGORICAL:
            vals = df[c].astype(str).to_numpy()
            vocab = self.categories_[c]
            block = np.zeros((len(df), len(vocab)), dtype=float)
            for j, v in enumerate(vocab):
                block[:, j] = (vals == v).astype(float)
            ohe_blocks.append(block)

        return np.hstack([num, *ohe_blocks])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        if not self.fitted_:
            raise RuntimeError("FeatureBuilder has not been fit.")
        return len(self.feature_names_)
