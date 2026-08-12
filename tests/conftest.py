"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from lead_scorer.sample_generator import generate_leads


@pytest.fixture(scope="session")
def leads_df() -> pd.DataFrame:
    """Small reproducible synthetic dataset for tests."""
    return generate_leads(n=800, seed=42)
