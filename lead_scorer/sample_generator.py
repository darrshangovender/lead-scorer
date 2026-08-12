"""Synthetic B2B leads dataset generator.

Produces a CSV that imitates the shape of a real CRM + web-engagement export:
firmographic columns (industry, headcount, region), behavioural columns
(page views, demo requests, email engagement), and a binary `converted` target.

The label is generated from a logistic noise model so that the relationship
between features and outcome is real but noisy — close to what you'd see in
production. Seeded for reproducibility.

Numbers calibrated so a well-tuned model lands at top-decile precision in the
0.45 - 0.6 range — comparable to (but not claiming) the production system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDUSTRIES: list[str] = [
    "saas",
    "fintech",
    "ecommerce",
    "logistics",
    "healthcare",
    "manufacturing",
    "education",
    "media",
]

REGIONS: list[str] = ["NA", "EMEA", "APAC", "LATAM", "AFRICA"]

SENIORITY_LEVELS: dict[str, int] = {
    "intern": 1,
    "individual_contributor": 2,
    "senior_ic": 3,
    "manager": 4,
    "director": 5,
    "vp": 6,
    "c_level": 7,
}


def _company_size_bucket(headcount: int) -> str:
    """Bucket raw headcount into the standard B2B sizing tiers."""
    if headcount < 10:
        return "micro"
    if headcount < 50:
        return "small"
    if headcount < 250:
        return "mid"
    if headcount < 1000:
        return "large"
    return "enterprise"


def generate_leads(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate `n` synthetic lead rows.

    Args:
        n: Number of leads to generate.
        seed: numpy random seed for reproducibility.

    Returns:
        DataFrame with raw lead columns plus a binary `converted` label.
    """
    rng = np.random.default_rng(seed)

    industry = rng.choice(INDUSTRIES, size=n, p=[0.22, 0.14, 0.16, 0.10, 0.12, 0.10, 0.08, 0.08])
    region = rng.choice(REGIONS, size=n, p=[0.45, 0.30, 0.12, 0.08, 0.05])

    # Headcount is heavy-tailed — log-normal is a decent approximation.
    headcount = np.clip(rng.lognormal(mean=4.0, sigma=1.6, size=n).astype(int), 1, 50_000)
    company_size = np.array([_company_size_bucket(h) for h in headcount])

    # Days since last touch — many recent, long tail of stale leads.
    days_since_last_touch = np.clip(rng.exponential(scale=18, size=n).astype(int), 0, 365)

    # Email engagement score in [0, 1] — beta gives a realistic skew.
    email_engagement_score = rng.beta(a=2.0, b=5.0, size=n)

    # Page views in last 30 days — Poisson with industry-driven rate.
    industry_rate = {
        "saas": 8.0,
        "fintech": 6.0,
        "ecommerce": 5.0,
        "logistics": 3.5,
        "healthcare": 3.0,
        "manufacturing": 2.5,
        "education": 4.0,
        "media": 4.5,
    }
    base_rate = np.array([industry_rate[i] for i in industry])
    page_views_30d = rng.poisson(lam=base_rate, size=n)

    # ~12% of leads request a demo. Strong positive signal.
    demo_requested = rng.binomial(n=1, p=0.12, size=n)

    seniority_labels = rng.choice(
        list(SENIORITY_LEVELS.keys()),
        size=n,
        p=[0.05, 0.30, 0.22, 0.18, 0.12, 0.08, 0.05],
    )
    seniority_score = np.array([SENIORITY_LEVELS[s] for s in seniority_labels])

    # --- Build conversion probability ------------------------------------
    # Real features carry the signal; noise term keeps it honest.
    logit = (
        -3.2
        + 1.6 * demo_requested
        + 0.10 * page_views_30d
        + 2.2 * email_engagement_score
        + 0.18 * seniority_score
        - 0.015 * days_since_last_touch
        + 0.35 * (company_size == "mid").astype(float)
        + 0.55 * (company_size == "large").astype(float)
        + 0.45 * (company_size == "enterprise").astype(float)
        + 0.30 * (industry == "saas").astype(float)
        + 0.20 * (industry == "fintech").astype(float)
        + rng.normal(loc=0.0, scale=0.6, size=n)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    converted = rng.binomial(n=1, p=p, size=n)

    df = pd.DataFrame(
        {
            "lead_id": np.arange(n),
            "industry": industry,
            "region": region,
            "headcount": headcount,
            "company_size": company_size,
            "days_since_last_touch": days_since_last_touch,
            "email_engagement_score": email_engagement_score.round(4),
            "page_views_30d": page_views_30d,
            "demo_requested": demo_requested,
            "seniority": seniority_labels,
            "seniority_score": seniority_score,
            "converted": converted,
        }
    )
    return df


def write_csv(path: str | Path, n: int = 5000, seed: int = 42) -> Path:
    """Generate and write a leads CSV. Returns the resolved output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_leads(n=n, seed=seed)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = write_csv("data/leads.csv")
    print(f"wrote {out} ({sum(1 for _ in open(out)) - 1} rows)")
