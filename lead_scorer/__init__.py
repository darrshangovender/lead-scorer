"""Lead Scorer — public reference implementation.

A B2B lead-ranking pipeline demonstrating the pattern used in production at
Agulhas Code (real client data under NDA). This package builds features from a
synthetic leads dataset, trains a logistic-regression baseline plus a
calibrated XGBoost ranker, and emits ranked scores.

Public API:
    Scorer          — orchestrates train + score with the production model
    FeatureBuilder  — feature engineering for raw lead rows
"""

from lead_scorer.features import FeatureBuilder
from lead_scorer.pipeline import Scorer

__all__ = ["FeatureBuilder", "Scorer"]
__version__ = "0.1.0"
