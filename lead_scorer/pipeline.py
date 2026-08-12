"""End-to-end orchestration: features -> train -> evaluate -> score.

The `Scorer` class is the entry point exposed at the package level. It runs
both models (LR and XGBoost) so the two-model interpretability story from the
production system is preserved here too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lead_scorer.evaluation import RankingMetrics, evaluate
from lead_scorer.features import FeatureBuilder
from lead_scorer.models import LRBaseline, XGBRanker

TARGET: str = "converted"


@dataclass
class TrainReport:
    """What `Scorer.train` returns — both models' metrics and feature names."""

    lr_metrics: RankingMetrics
    xgb_metrics: RankingMetrics
    feature_names: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr_metrics.as_dict(),
            "xgb": self.xgb_metrics.as_dict(),
            "feature_names": self.feature_names,
        }


@dataclass
class Scorer:
    """Trains both models, evaluates on a holdout, then scores new leads."""

    test_size: float = 0.25
    random_state: int = 42

    builder_: FeatureBuilder = field(default_factory=FeatureBuilder)
    lr_: LRBaseline = field(default_factory=LRBaseline)
    xgb_: XGBRanker = field(default_factory=XGBRanker)
    trained_: bool = False

    def train(self, df: pd.DataFrame) -> TrainReport:
        """Fit both models on a stratified split of `df`. Returns metrics."""
        if TARGET not in df.columns:
            raise ValueError(f"Training frame must contain {TARGET!r} column.")

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[TARGET],
        )

        X_train = self.builder_.fit_transform(train_df)
        X_test = self.builder_.transform(test_df)
        y_train = train_df[TARGET].to_numpy()
        y_test = test_df[TARGET].to_numpy()

        self.lr_.fit(X_train, y_train)
        self.xgb_.fit(X_train, y_train)
        self.trained_ = True

        return TrainReport(
            lr_metrics=evaluate(y_test, self.lr_.predict_proba(X_test)),
            xgb_metrics=evaluate(y_test, self.xgb_.predict_proba(X_test)),
            feature_names=list(self.builder_.feature_names_),
        )

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score new leads with the production (XGBoost) model.

        Returns the input frame with an added `score` column, sorted descending.
        Includes the LR score too as `score_baseline` for the interpretability
        comparison.
        """
        if not self.trained_:
            raise RuntimeError("Scorer.score called before train().")
        X = self.builder_.transform(df)
        scored = df.copy()
        scored["score_baseline"] = self.lr_.predict_proba(X)
        scored["score"] = self.xgb_.predict_proba(X)
        return scored.sort_values("score", ascending=False).reset_index(drop=True)

    def score_to_json(self, df: pd.DataFrame, path: str | Path, top_n: int | None = None) -> Path:
        """Score `df` and write top-N (or all) ranked leads to a JSON file."""
        scored = self.score(df)
        if top_n is not None:
            scored = scored.head(top_n)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        records = scored.to_dict(orient="records")
        out.write_text(json.dumps(records, indent=2, default=str))
        return out


def run_from_csv(
    leads_csv: str | Path,
    output_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """CLI-style entry point: load CSV, train, evaluate, write metrics.

    Used by `python -m lead_scorer.pipeline` and the Makefile's `make eval` target.
    """
    df = pd.read_csv(leads_csv)
    scorer = Scorer()
    report = scorer.train(df)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report.as_dict(), indent=2))

    # Score the full dataset (in production you'd score only newly-arrived leads).
    scorer.score_to_json(df.drop(columns=[TARGET]), out_dir / "scored_leads.json", top_n=100)
    return report.as_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--leads", default="data/leads.csv")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    metrics = run_from_csv(args.leads, args.out)
    print(json.dumps(metrics, indent=2))
