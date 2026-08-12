"""Tests for FeatureBuilder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_scorer.features import CATEGORICAL, NUMERIC, RAW_REQUIRED, FeatureBuilder


def test_required_columns_present(leads_df: pd.DataFrame) -> None:
    for col in RAW_REQUIRED:
        assert col in leads_df.columns, f"sample generator dropped {col}"


def test_no_nulls_in_required_fields(leads_df: pd.DataFrame) -> None:
    for col in RAW_REQUIRED:
        assert not leads_df[col].isna().any(), f"nulls found in {col}"


def test_fit_then_transform_shape(leads_df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    X = fb.fit_transform(leads_df)
    expected_cols = len(NUMERIC) + sum(len(fb.categories_[c]) for c in CATEGORICAL)
    assert X.shape == (len(leads_df), expected_cols)
    assert fb.n_features == expected_cols
    assert len(fb.feature_names_) == expected_cols


def test_transform_before_fit_raises() -> None:
    fb = FeatureBuilder()
    with pytest.raises(RuntimeError):
        fb.transform(pd.DataFrame())


def test_missing_column_raises(leads_df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    bad = leads_df.drop(columns=["industry"])
    with pytest.raises(ValueError, match="Missing required"):
        fb.fit(bad)


def test_null_in_required_raises(leads_df: pd.DataFrame) -> None:
    fb = FeatureBuilder()
    bad = leads_df.copy()
    bad.loc[0, "industry"] = None
    with pytest.raises(ValueError, match="nulls"):
        fb.fit(bad)


def test_one_hot_sums_to_one_per_categorical(leads_df: pd.DataFrame) -> None:
    """Each categorical block should be a proper one-hot — sums to 1 per row."""
    fb = FeatureBuilder().fit(leads_df)
    X = fb.transform(leads_df)
    offset = len(NUMERIC)
    for c in CATEGORICAL:
        width = len(fb.categories_[c])
        block = X[:, offset : offset + width]
        np.testing.assert_array_equal(block.sum(axis=1), np.ones(len(leads_df)))
        offset += width


def test_unseen_category_falls_through_as_zero(leads_df: pd.DataFrame) -> None:
    """Categories not seen at fit time produce all-zero rows for that block."""
    fb = FeatureBuilder().fit(leads_df)
    novel = leads_df.head(1).copy()
    novel["industry"] = "quantum_widgets"
    X = fb.transform(novel)
    # Find the industry block.
    offset = len(NUMERIC)
    width = len(fb.categories_["industry"])
    block = X[:, offset : offset + width]
    assert block.sum() == 0
